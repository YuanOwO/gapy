# -*- coding: utf-8 -*-

import csv
import json
import os

import numpy as np
import yaml

from ga2 import *
from ga.display import plot_init

config, data = None, None

# numpy 顯示設定
np.set_printoptions(linewidth=120, formatter={"complex_kind": "{:.4f}".format, "float_kind": "{:.4f}".format})


def parse_config() -> None:
    """
    解析設定檔
    """

    global config, data
    # 載入設定檔
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 處理城市資料
    dcf = config["data"]
    if dcf["from_file"]:  # 從檔案讀取資料
        fmt = dcf["file"]["format"]
        path = dcf["file"]["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到檔案: {path}")
        with open(path, "r", encoding="utf-8") as f:
            if fmt == "csv":
                reader = csv.reader(f)
                data = []
                for row in reader:
                    data.append(row)
                del reader
            elif fmt == "json":
                data = json.load(f)
            else:
                raise ValueError(f"不支援的檔案格式: {fmt}")
    else:  # 產生隨機資料
        size = dcf["random"]["size"]
        low, high = dcf["random"]["min"], dcf["random"]["max"]
        data = rand.uniform(low, high, (size, 2))

    data = parse_location(data)  # 將城市資料轉換成複數形式

    # 設定人口數
    mode = config["ga"]["population"]["mode"]
    if mode == "value":  # 人口數為固定值
        config["ga"]["population"] = config["ga"]["population"]["value"]
    elif mode == "ratio":  # 人口數為資料點數的比例
        config["ga"]["population"] = len(data) * config["ga"]["population"]["value"]

    # 設定繪圖
    plot_init(config["plot"])


if __name__ == "__main__":
    parse_config()

    with GA2(data, **config["ga"]) as ga:
        ga.run()

        history = ga.history

    #     ga.output_result()
    # ga = GA(data, **config["ga"])
    # ga.run()
    # ga.output_result()
