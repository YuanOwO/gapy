from hashlib import sha256
from json import JSONEncoder
from sys import version

import numpy as np
from numpy import double
from numpy.random import MT19937, Generator

# seed = int(datetime.now().timestamp() * 1000)
seed = sha256(version.encode()).digest()
seed = int.from_bytes(seed, "big")
mt = MT19937(seed)
rand = Generator(mt)


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

        if isinstance(obj, double):
            return str(obj)

        return JSONEncoder.default(self, obj)
