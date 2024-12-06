import json
import multiprocessing as mp
import os
import shutil
from multiprocessing.sharedctypes import RawArray

import numpy as np
from numpy.ctypeslib import as_ctypes_type
from numpy.typing import NDArray
from tqdm import tqdm

from .logger import LogListener, get_logger
from .utils import AdvancedJSONEncoder
from .worker import Worker


class GA2:
    """
    遺傳演算法 (Genetic Algorithm, GA)

    用於解決旅行商問題 (Travelling Salesman Problem, TSP)
    """

    def __init__(
        self,
        locations: NDArray[np.cdouble],
        generations: int = None,
        population: int = None,
        k: int = None,
        crossover_rate: float = None,
        mutation_rate: float = None,
        **kwargs,
    ) -> None:
        """
        基因演算法 (Genetic Algorithm, GA)

        Args:
            locations (list[Location]): 所有城市的座標
            generations (int):          要演化幾代，預設為 100 代
            population (int):           群體的大小，預設為城市數量的 100 倍
            k (int):                    基因要切成幾段，預設為 2 段
            crossover_rate (float):     表現最好的前幾名才能夠交配，值必須在 [0, 1] 之間，預設為 0.8 (80%)
            mutation_rate (float):      基因突變率，值必須在 [0, 1] 之間，預設為 0.1 (10%)
        """
        self.log_queue = mp.Queue()
        self.log = get_logger(__name__, self.log_queue)

        self.locations = locations
        self.generations = generations or 100

        n = locations.shape[0]  # 城市數量
        self.population = population or n * 100
        self.k = min(k or 2, n)  # 每段基因至少有一個城市
        self.crossover_rate = crossover_rate or 0.8
        self.mutation_rate = mutation_rate or 0.1

        self.log.info(f"城市數量: {n}")
        self.log.info(f"演化代數: {self.generations}")
        self.log.info(f"群體大小: {self.population}")
        self.log.info(f"基因切割: {self.k}")
        self.log.info(f"交配率: {self.crossover_rate}")
        self.log.info(f"突變率: {self.mutation_rate}")

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.best = None  # 歷史最佳路徑
        self.history = []  # 每一代的歷史紀錄

        self.workers_num = mp.cpu_count()  # 使用 CPU 核心數量作為工作進程數量
        self.chunk_size = population // self.workers_num + 1  # 每個工作進程處理的任務數量
        while self.chunk_size >= 1024:
            self.chunk_size //= 8
        self.workers = []

        self.task_queue = mp.JoinableQueue()
        self.result_queue = mp.Queue()
        self.lock = mp.Lock()

        # 建立共享記憶體
        p = int(self.population * self.crossover_rate)  # 可以交配的個體數量
        self._cross_num = p
        self.raw_genes = RawArray(as_ctypes_type(np.int64), p * n)  # 儲存基因的共享記憶體
        self.raw_p = RawArray(as_ctypes_type(np.float64), p)  # 儲存機率的共享記憶體
        self.lock = mp.Lock()
        self.workers = []

        self.log.info(f"Worker 數量: {self.workers_num}")
        self.log.info(f"Chunk 大小: {self.chunk_size}")
        self.log.info(f"可交配的個體數量: {p}")

        self.logListener = LogListener(self.log_queue)
        self.logListener.start()
        self.log.info("-" * 50)

        # 結果資料夾
        shutil.rmtree("results", ignore_errors=True)
        os.makedirs("results/history", exist_ok=True)

    def __enter__(self):
        self.start_workers()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log.info(f"GA terminated")
        self.stop_workers()
        self.logListener.terminate()

    def start_workers(self):
        """
        啟動所有工作進程
        """
        self.log.info(f"Starting GA workers")
        data_buf = self.locations.tobytes()
        for _ in range(self.workers_num):  # 創建工作進程
            worker = Worker(
                self.task_queue,
                self.result_queue,
                self.log_queue,
                self.raw_genes,
                self.raw_p,
                self.lock,
                data_buf,
                self._cross_num,
                self.k,
                self.mutation_rate,
            )
            worker.start()
            self.workers.append(worker)
        self.log.info("-" * 50)

        cnt = 0
        while cnt < self.workers_num:  # 等待所有工作進程啟動
            self.result_queue.get()
            cnt += 1

    def stop_workers(self):
        """
        結束所有工作進程
        """
        self.log.info(f"Stopping GA workers")
        for _ in range(self.workers_num):
            self.task_queue.put(None)  # 傳送結束訊息
        self.task_queue.join()

    def run(self):
        """
        啟動演化
        """
        population, chunk_size = self.population, self.chunk_size

        individuals = []
        for gen in tqdm(range(self.generations), desc="演化進度"):
            self.log.info(f"開始演化第 {gen}/{self.generations} 代")

            if gen == 0:  # 第一代隨機產生
                self.log.info(f"隨機產生")
                for i in range(0, population, chunk_size):  # 產生初始化任務
                    self.task_queue.put(("init", min(chunk_size, population - i)))
            else:  # 第二代後由上一代產生
                self.log.info(f"交配產生")
                p = int(population * self.crossover_rate)  # 可以交配的個體數量
                threshold = np.quantile(dis, self.crossover_rate)  # 取得可以交配的門檻值
                # 篩選表現好的個體
                mask = dis <= threshold
                gene, dis = gene[mask], dis[mask]
                gene, dis = gene[:p], dis[:p]  # 確保長度固定

                fitness = np.round(1 / dis, 7)  # 計算適應度
                p = fitness / np.sum(fitness, dtype=np.float64)  # 個體被選中的機率

                # 更新共享記憶體
                self.raw_genes[:] = gene.flatten()
                self.raw_p[:] = p
                for _ in range(self.workers_num):
                    self.task_queue.put(("update", None))
                for _ in range(self.workers_num):  # 等待更新完成
                    self.result_queue.get()

                for i in range(0, population, chunk_size):  # 產生交配任務
                    self.task_queue.put(("crossover", min(chunk_size, population - i)))

            # 取得結果 (個體)
            individuals.clear()
            desc = "擇偶中" if gen > 0 else "初始化中"
            with tqdm(total=population, desc=desc, leave=False, position=1) as pbar:
                for _ in range(0, population, chunk_size):
                    ret = self.result_queue.get()
                    individuals.extend(ret)
                    pbar.update(len(ret))

            self.log.debug(f"第 {gen} 代個體生成完成，數量: {len(individuals)}")

            gene, dis = zip(*individuals)  # 個體的基因(路徑)和距離
            gene = np.array([np.frombuffer(g, dtype=np.int64) for g in gene], dtype=np.int64)
            dis = np.array(dis, dtype=np.float64)

            best_idx = np.argmin(dis)
            best = gene[best_idx]
            if self.best is None or dis[best_idx] < self.best[1]:
                self.best = [best, dis[best_idx]]

            self.log.info(f"第 {gen} 代最佳距離: {self.best[1]}")

            with open(f"results/history/{gen:03d}.json", "w", encoding="utf-8") as f:
                data = {"generation": gen, "best": best.tolist(), "data": dis}
                json.dump(
                    data,
                    f,
                    cls=AdvancedJSONEncoder,
                )

            self.history.append(
                {
                    "generation": gen,
                    "best": best,
                    # "summary": {
                    #     "mean": np.mean(dis),
                    #     "std": np.std(dis),
                    #     "min": np.min(dis),
                    #     "q1": np.percentile(dis, 25),
                    #     "median": np.median(dis),
                    #     "q3": np.percentile(dis, 75),
                    #     "max": np.max(dis),
                    # },
                }
            )

            self.log.info("-" * 50)

        self.log.info(f"GA finished")
        with open("results/history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, cls=AdvancedJSONEncoder)
