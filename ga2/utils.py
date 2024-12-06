import os
from collections.abc import Hashable
from datetime import datetime
from hashlib import sha256
from json import JSONEncoder
from sys import version

import numpy as np
from numpy.random import MT19937, Generator
from numpy.typing import ArrayLike, NDArray


def parse_location(data: ArrayLike) -> NDArray[np.cdouble]:
    """
    將資料轉換成座標

    Args:
        data (ArrayLike): 資料

    Returns:
        NDArray[np.cdouble]: 座標
    """
    ret = np.array([np.cdouble(*d) for d in data])
    return ret


def generator(seed):
    """
    根據種子 (`seed`) 產生一個使用 MT19937 演算法的隨機數生成器。

    其中 `seed` 可以是任意的東西：
    -   `int`: 直接使用該數字當作種子
    -   `bytes`: 使用 sha256 進行雜湊轉換成 `int`
    -   `str`: 使用 UTF-8 編碼轉換成 `bytes` 後再執行上述動作
    -   `datetime`: 使用其時間戳記 (單位為毫秒) 轉換成 `int`
    -   其他 `hashable` 物件: 使用 `hash()` 轉換成 `int`
    -   其他物件: 拋出 `ValueError` 錯誤
    """

    if seed is None:
        seed = os.urandom(16)  # 使用隨機位元組作為種子

    if isinstance(seed, str):
        seed = seed.encode()
    if isinstance(seed, bytes):
        seed = int.from_bytes(sha256(seed).digest(), "big", signed=False)

    if isinstance(seed, datetime):
        seed = int(seed.timestamp() * 1000)

    if not isinstance(seed, int):
        if isinstance(seed, Hashable):
            seed = hash(seed)
        else:
            raise ValueError(f"Unsupported type: {type(seed)}")

    if seed < 0:  # seed 必須為正整數 => 取絕對值
        seed = -seed

    mt = MT19937(seed)
    return Generator(mt)


# 預設的隨機數生成器
rand = generator(version)  # 使用 Python 版本作為種子


class AdvancedJSONEncoder(JSONEncoder):
    """
    進階的 JSON 編碼器，支援自訂物件的序列化。

    這個類別繼承自 `json.JSONEncoder`，並且覆寫了 `default` 方法，  
    當遇到無法序列化的物件時，會先檢查是否有 `__json__()` 方法，  
    如果有的話就使用該方法回傳的值，否則就使用原本的方法處理 (通常會拋出 `TypeError`)。
    若遇到 `numpy.ndarray` 物件時，會先轉換成 `list` 再進行序列化。
    """

    def default(self, obj):
        if hasattr(obj, "__json__"):
            return obj.__json__()
        # print
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.inexact):  # numpy 的浮點數型別 (包含複數)
            return str(obj)

        return JSONEncoder.default(self, obj)
