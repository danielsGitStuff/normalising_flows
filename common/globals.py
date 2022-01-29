import sys

from common.jsonloader import Ser
from typing import Any

import numpy as np

from common import util
from common.poolreplacement import RestartingPoolReplacement

pool_key = "pool//instance"

class Global:
    class Testing:
        """This is for the crazy stuff like injecting code where it normally does not belong. Use with care!"""

        @staticmethod
        def set(key: str, value: Any):
            d = Global.Testing.d
            new_key: str = key
            if not key.startswith('testing_'):
                new_key = f"testing_{key}"
                print(f"testing key '{key}' does not start with 'testing_'. I change it to '{new_key}'.", file=sys.stderr)
            d[new_key] = value

        @staticmethod
        def has(key: str) -> bool:
            d = Global.Testing.d
            new_key: str = key
            if not key.startswith('testing_'):
                new_key = f"testing_{key}"
                print(f"testing key '{key}' does not start with 'testing_'. I change it to '{new_key}'.", file=sys.stderr)
            return new_key in d

        @staticmethod
        def get(key: str, default_value: Any = None) -> Any:
            d = Global.Testing.d
            new_key: str = key
            if not key.startswith('testing_'):
                new_key = f"testing_{key}"
                print(f"testing key '{key}' does not start with 'testing_'. I change it to '{new_key}'.", file=sys.stderr)
            return d.get(new_key, default_value)

        @staticmethod
        def set_global(key: str, value: Any):
            d = Global.Testing.d
            if key in d:
                util.eprint(f"CHANGING GLOBAL VARIABLE '{key}' from '{d[key]}' to '{value}'")
            else:
                util.eprint(f"SETTING GLOBAL VARIABLE '{key}' to '{value}'")
            d[key] = value

    @staticmethod
    def equals(key, value: Any) -> bool:
        d = Global.d
        if key in d:
            v = d[key]
            return v == value
        return False

    @staticmethod
    def set_global(key: str, value: Any):
        d = Global.d
        if key in d:
            util.eprint(f"CHANGING GLOBAL VARIABLE '{key}' from '{d[key]}' to '{value}'")
        else:
            util.eprint(f"SETTING GLOBAL VARIABLE '{key}' to '{value}'")
        d[key] = value

    @staticmethod
    def get_default(key: str, default_value: Any) -> Any:
        d = Global.d
        if key in d:
            return d[key]
        print(f"key '{key}' was not in Globals. Returning '{default_value}'", file=sys.stderr)
        return default_value

    @staticmethod
    def set_computation_pool_size(size: int):
        Global.set_global("computation_pool_size", size)

    @staticmethod
    def GET_MAX() -> int:
        # todo remove
        print("method Global.GET_MAX() is deprecated", file=sys.stderr)
        return Global.d["global_max"]

    @staticmethod
    def SET_MAX(max: int):
        Global.set_global("global_max", max)

    @staticmethod
    def get_generator_xs_dtype() -> np.dtype:
        return Global.d["gen_xs_dtype"]

    @staticmethod
    def set_generator_xs_dtype(dtype: np.dtype):
        Global.set_global("gen_xs_dtype", dtype)

    @staticmethod
    def WIDTH_TEAMS():
        return Global.d["width_teams"]

    @staticmethod
    def WIDTH_TIME():
        return Global.d["width_time"]

    @staticmethod
    def POOL() -> RestartingPoolReplacement:
        if pool_key not in Global.d:
            Global.set_global(pool_key, RestartingPoolReplacement(Global.d["computation_pool_size"]))
        pool: RestartingPoolReplacement = Global.d[pool_key]
        return pool

    @staticmethod
    def get_noise_decline_start() -> int:
        return Global.d["noise_decline_start"]

    @staticmethod
    def get_noise_decline_stop() -> int:
        return Global.d["noise_decline_stop"]

    @staticmethod
    def set_seed(seed_value: int):
        for _ in range(1):
            print(f"FIXING SEED TO {seed_value}", file=sys.stderr)
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        # import os
        # os.environ['PYTHONHASHSEED'] = str(seed_value)

        # 2. Set `python` built-in pseudo-random generator at a fixed value
        import random
        random.seed(seed_value)

        # 3. Set `numpy` pseudo-random generator at a fixed value
        import numpy as np
        np.random.seed(seed_value)

        # 4. Set the `tensorflow` pseudo-random generator at a fixed value
        import tensorflow as tf
        tf.random.set_seed(seed_value)


Global.d = {"global_max": 1, "gen_xs_dtype": np.float, "width_teams": 5, "width_time": 4, "noise_decline_start": 100, "noise_decline_stop": 180, "computation_pool_size": 8}
Global.Testing.d = {}
# for _ in range(5):
#     print('ENABLING KALEIDO_MISSING_HACK: no 3d prints!!!', file=sys.stderr)
# Global.Testing.set('kaleido_missing_hack', True)  # currently no kaleido for python 3.10
