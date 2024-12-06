import logging
import multiprocessing as mp
from itertools import chain

import numpy as np

from .logger import get_logger
from .utils import generator


class Worker(mp.Process):
    def __init__(
        self,
        task_queue: mp.JoinableQueue,
        result_queue: mp.Queue,
        log_queue: mp.Queue,
        raw_genes,
        raw_p,
        lock: mp.Lock,
        locations: bytes,
        cross_num: int,
        k: int,
        mutation_rate: float,
    ):
        """
        Args:
            task_queue (mp.Queue): 接收任務的 Queue
            result_queue (mp.Queue): 傳送結果的 Queue
        """
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.log_queue = log_queue
        self.raw_genes = raw_genes
        self.raw_p = raw_p
        self.lock = lock
        self.locations = np.frombuffer(locations, dtype=np.cdouble)
        self.n = self.locations.shape[0]
        self.cross_num = cross_num
        self.k = k
        self.mutation_rate = mutation_rate

    def run(self):
        self.log = get_logger(self.name, self.log_queue)
        self.log.info("Worker 開始")

        # seed = self.pid  # 使用當前進程的 PID 作為種子
        seed = None  # urandom 作為種子
        self.rand = generator(seed)  # 產生隨機數生成器
        # self.log.debug(f"設定 random 種子為 {seed}")

        self.result_queue.put("hola")  # 進程已啟動

        while True:
            task = self.task_queue.get()  # 取得任務
            self.log.debug(f"收到任務: {task}")

            if task is None:  # 收到 None 表示結束
                self.task_queue.task_done()  # 標記任務完成
                break

            ####################################################################################################

            cmd, arg = task  # 拆解任務指令
            ret = None  # 預設回傳值

            if cmd == "init":  # 初始化
                self.log.debug(f"初始化 {arg}")
                ret = [self.shuffle() for _ in range(arg)]

            ##################################################

            elif cmd == "update":  # 更新資料
                self.log.debug(f"更新資料")

                self.gene = np.frombuffer(self.raw_genes, dtype=np.int64).reshape(-1, self.n)
                self.p = np.frombuffer(self.raw_p, dtype=np.float64)

                self.idx = np.arange(self.cross_num)  # 索引

                if self.gene.shape[0] != self.cross_num:  # 基因數量不對
                    self.log.error(f"基因數量不對: {self.gene.shape[0]} != {self.cross_num}")
                if self.p.shape[0] != self.cross_num:  # 機率數量不對
                    self.log.error(f"機率數量不對: {self.p.shape[0]} != {self.cross_num}")
                if np.sum(self.p) != 1:  # 機率總和不為 1
                    self.log.error(f"機率總和不為 1: {np.sum(self.p)}")
                    # 修正精度後再次計算
                    self.p = np.round(self.p, 7)
                    self.p /= np.sum(self.p, dtype=np.float64)

            ##################################################

            elif cmd == "crossover":  # 交配
                self.log.debug(f"交配 {arg}")
                ret = [self.crossover() for _ in range(arg)]

            ##################################################

            else:
                self.log.error(f"未知指令: {cmd}")

            self.result_queue.put(ret)  # 傳送結果
            self.task_queue.task_done()  # 標記任務完成
            self.log.debug(f"任務完成: {task}")

        self.log.info("Worker 終止")

    def calc_distance(self, path):
        """
        計算路徑的總距離

        Args:
            path (np.ndarray): 路徑
        """
        loc = self.locations[path]  # 城市座標
        diff = np.diff(loc, append=loc[0])  # 計算城市間的距離
        return np.sum(np.abs(diff))  # 回傳總距離

    def shuffle(self):
        """
        隨機排序城市的走訪順序，並計算此路徑的總距離。
        其中 0 號城市為起點，一定會在第一個位置。
        """
        # 隨機排序城市的走訪順序
        # path 存的是城市的索引，從 0 開始
        path = self.rand.permutation(np.arange(1, self.n))  # 隨機排序城市
        path = np.insert(path, 0, 0)  # 加入起點

        # 計算總距離
        dis = self.calc_distance(path)

        if path.shape[0] != self.n:
            self.log.error(f"基因長度錯誤: {path.shape[0]} != {self.n}")

        buf = path.tobytes()  # 將路徑轉換成 bytes
        return buf, dis

    def crossover(self):
        """
        交配
        """

        f1, f2 = self.rand.choice(self.idx, 2, p=self.p)  # 隨機選擇兩個父母
        f1, f2 = self.gene[[f1, f2]]  # 取得父母的基因

        # 將 F1 切成 k 段，取偶數段
        g1 = np.array_split(f1, self.k)[::2]
        _tmp = np.concatenate(g1)

        # F2 取不在 F1 的城市
        g2 = np.setdiff1d(f2, _tmp, assume_unique=True)
        g2 = np.array_split(g2, self.k // 2)  # 將 F2 切成 k / 2 段

        # 交錯排列合併基因
        ch = list(chain.from_iterable(zip(g1, g2)))

        if len(g1) > len(g2):  # g1 比 g2 長 (k 為奇數)
            ch.append(g1[-1])

        ch = np.concatenate(ch)

        if ch.shape[0] != self.n:  # 長度不對
            self.log.error(f"基因長度錯誤: {ch.shape[0]} != {self.n}")
            raise ValueError(f"Invalid length of gene: {ch.shape[0]}, expected {self.n}")

        # 突變
        n = self.n  # 城市數量
        for i, r in np.ndenumerate(self.rand.random(n)):
            i = i[0]
            if i == 0:  # 起點不能變
                continue
            if r <= self.mutation_rate:  # 發生突變
                j = self.rand.integers(1, n)
                ch[[i, j]] = ch[[j, i]]

        # 計算總距離
        dis = self.calc_distance(ch)

        buf = ch.tobytes()  # 將路徑轉換成 bytes
        return buf, dis
