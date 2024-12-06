from itertools import combinations

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import double

from .location import Route

__all__ = ["plot_init", "plot_route", "plot_summary"]

# 路徑的樣式
possible_path_style = dict(color="k", alpha=0.5)
now_path_style = dict(color="#d62728")
best_path_style = dict(color="#1f77b4")
figsize = {}


def plot_init(config) -> None:
    """
    初始化設定
    """
    # 加入字型
    font = fm.FontEntry(fname=config["font"]["path"], name=config["font"]["name"])
    fm.fontManager.ttflist.append(font)

    # 設定圖片參數
    mpl.rcParams.update(**config["rcParams"])
    mpl.rcParams["font.family"] = font.name

    del config["rcParams"], config["font"]

    globals().update(config)


def plot_route(filename: str, generation: int, route: Route, best: Route = None) -> None:
    fig = plt.figure(figsize=figsize["route"])
    ax = fig.subplots()

    # 取得所有座標的 x, y
    x: tuple[double]
    y: tuple[double]
    x, y = zip(*([loc.coord for loc in route] + [route[0].coord]))
    min_x, max_x = np.floor(np.min(x)), np.ceil(np.max(x))
    min_y, max_y = np.floor(np.min(y)), np.ceil(np.max(y))
    dx, dy = (max_x - min_x) / 10, (max_y - min_y) / 10

    # 設定標題、座標軸
    ax.set_title(f"GA 第 {generation} 代的最佳路徑")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks(np.arange(min_x, max_x + dx, dx))
    ax.set_xticks(np.arange(min_x, max_x + dx, dx / 5), minor=True)
    ax.set_yticks(np.arange(min_y, max_y + dy, dy))
    ax.set_yticks(np.arange(min_y, max_y + dy, dy / 5), minor=True)

    # 畫出其他可以走的路徑 (很花時間)
    # for loc1, loc2 in combinations(route, 2):
    #     ax.plot(*zip(loc1.coord, loc2.coord), **possible_path_style)

    # 畫出走訪的路徑
    ax.plot(x, y, **now_path_style, label=f"當代最佳路徑 ({route.distance:.4f})", zorder=11)

    # 畫出最佳路徑
    if best is not None:
        bx, by = zip(*([loc.coord for loc in best] + [best[0].coord]))
        ax.plot(bx, by, **best_path_style, label=f"歷史最佳路徑 ({best.distance:.4f})", zorder=10)

    # 標記走訪的順序
    for i, loc in enumerate(route):
        coord = (loc.x, loc.y + dy / 10)  # 文字高一點
        ax.annotate(i, coord, zorder=12)

    # 標記總長
    coord = (max_x - dx / 5, max_y - dy / 5)
    # ax.annotate(
    #     f"總長: {route.distance:.4f}",
    #     coord,
    #     ha="right",
    #     va="top",
    #     bbox=dict(boxstyle="round,pad=0.5", facecolor="w", alpha=0.5),
    # )
    ax.legend()

    fig.savefig(filename)  # 儲存圖片
    plt.close(fig)  # 關閉圖片


def plot_summary(filename: str, summary: dict) -> None:
    mean = np.array(summary["mean"])
    std = np.array(summary["std"])
    min_ = np.array(summary["min"])
    q1 = np.array(summary["q1"])
    median = np.array(summary["median"])
    q3 = np.array(summary["q3"])
    max_ = np.array(summary["max"])

    fig = plt.figure(figsize=figsize["summary"])
    ax = fig.subplots()

    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1)

    # 設定標題、座標軸
    ax.set_title("GA 演化過程")
    ax.set_xlabel("世代數")
    ax.set_ylabel("距離")
    max_x, min_y, max_y = len(mean), np.min(min_) * 0.95, np.max(max_) * 1.1
    best_x = np.argmin(min_)
    best_y = min_[best_x]
    ax.set_xlim(0, max_x - 1)
    # ax.set_ylim(min_y, max_y)
    ax.set_xticks(np.arange(0, max_x, max(max_x // 10, 1)))
    ax.set_xticks(np.arange(0, max_x, max(max_x // 50, 1)), minor=True)
    # ax.set_yticks(np.arange(min_y, max_y, (max_y - min_y) / 5))
    # ax.set_yticks(np.arange(min_y, max_y, (max_y - min_y) / 10), minor=True)
    ax.yaxis.set_major_locator(MaxNLocator(10, steps=[1, 2, 4, 5, 10]))
    ax.yaxis.set_minor_locator(MaxNLocator(50, steps=[1, 2, 4, 5, 10]))
    ax.grid()

    # 畫出演化過程
    ax.plot(median, label="中位數")
    ax.plot(min_, label="最短距離")
    ax.plot(max_, label="最長距離")
    plt.fill_between(range(len(mean)), q1, q3, alpha=0.5, label="IQR")

    ax.plot(best_x, best_y, "o")
    ax.annotate(
        f"最佳解: {best_y:.4f} (在第 {best_x} 代)",
        (best_x, best_y),
        (best_x + max_x / 20, best_y - 0.1),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.legend()
    fig.savefig(filename)  # 儲存圖片
    plt.close(fig)  # 關閉圖片
