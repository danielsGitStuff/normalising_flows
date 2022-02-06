from __future__ import annotations
import io
import json
import os
import sys
from abc import ABC, ABCMeta
from datetime import date
from importlib import import_module
from numbers import Number
from pathlib import Path
from typing import Set, Tuple, List, Dict, Type

import numpy as np

from common import util

# from GPy.models import GPRegression
SerSettings = object()
SerSettings.__testing_mode = False


class Ser(ABC):

    def __init__(self):
        self.k__ = self.__class__.__qualname__
        self.m__ = self.__class__.__module__
        if not Ser.__testing_mode and self.m__ == '__main__':
            raise RuntimeError(
                f"class '{self.k__}' was instantiated in your '__main__' module. This way it can only be deserialised in this very module (whatever '__main__' refers to right now) again. It is impossible to find the correct absolute module name for now.")
        self.dates: Set[str] = set()
        self.datetimes: Set[str] = set()
        self.ignored: Set[str] = {"ignored", "dates", "datetimes", "ser_deserialize_lazy", "ser_deserialize_lazy_z"}
        # if True this object is not normally deserialized (not into Ser objects, just Python/Json-objects).
        #  call ser_eager_load(). This may speed loading up by a lot
        self.ser_deserialize_lazy: bool = False
        self.ser_deserialize_lazy_z = None

    def ser_eager_load(self):
        if self.ser_deserialize_lazy:
            ser_instanciate(self.ser_deserialize_lazy_z, instance=self, eager=True)
            self.ser_deserialize_lazy = False
            self.ser_deserialize_lazy_z = None
        return self

    def to_json(self, file: str = None, pretty_print: bool = False):
        return to_json(self, file=file, pretty_print=pretty_print)

    def jsonize(self):
        return load_json(to_json(self))

    def after_deserialize(self):
        """you may rearrange or cast some values after the instance has been deserialized"""
        pass

    def unwrap(self):
        """in case your instance just wraps another one you can replace the wrapper here by returning its wrapped content"""
        return self

    @staticmethod
    def enable_testing():
        for _ in range(10):
            print('SERIALISATION TESTING MODE ENABLED. You can create Ser objects in the __main__ module but deserialisation will break!', file=sys.stderr)
        SerSettings.__testing_mode = True


def write_text_file(path, text):
    f = io.open(path, mode="w", encoding="utf-8")
    f.write(text)
    f.close()


class NP(Ser):
    def __init__(self):
        super().__init__()
        self.shape: Tuple[int] = None
        self.array: List[Number] = None
        self.dtype: str = None

    def unwrap(self):
        dtype = np.dtype(self.dtype)
        shape = tuple(self.shape)
        array = np.array(self.array, dtype=dtype).reshape(shape)
        return array

    @staticmethod
    def wrap(ar: np.ndarray) -> NP:
        n = NP()
        n.dtype = str(ar.dtype)
        n.array = ar.tolist()
        n.shape = ar.shape
        return n


class TW(Ser):
    """wraps Type"""

    def __init__(self):
        super().__init__()
        self.module: str = None
        self.klass: str = None

    @staticmethod
    def wrap(t: type) -> TW:
        tw = TW()
        tw.module = t.__module__
        tw.klass = t.__name__
        return tw

    def unwrap(self) -> type:
        mod = import_module(self.module)
        klass = getattr(mod, self.klass)
        return klass


class PW(Ser):
    """Wraps Path"""

    def __init__(self):
        super().__init__()
        self.p: str = None

    def unwrap(self):
        return Path(self.p)

    @staticmethod
    def wrap(p: Path) -> PW:
        pw = PW()
        pw.p = str(p)
        return pw


def load_json(file: [Path, str], raise_on_404: bool = False):
    if os.path.exists(file):
        try:
            with open(file, encoding='utf8') as json_file:
                js = json_file.read()
                return from_json(js)
        except Exception as e:
            util.eprint(f"got {type(e)}")
            if len(e.args) > 0:
                util.eprint(e.args[0])
            util.eprint("could not deserialize '{}' for some reason".format(file))
            raise e
    elif raise_on_404:
        raise FileNotFoundError(f"could not find file '{file}'")


def __is_klass(deserialized):
    return isinstance(deserialized, dict) and "k__" in deserialized


def ser_instanciate(deserialized, instance: Ser = None, eager: bool = False):
    if __is_klass(deserialized):
        if instance is None:
            instance = __new_instance(deserialized["k__"], deserialized["m__"])
        if instance.ser_deserialize_lazy:
            if not eager:
                instance.ser_deserialize_lazy_z = deserialized
                return instance
        for k, v in deserialized.items():
            if k == "k__" or k == "m_" or k == "dates":
                continue
            val = ser_instanciate(v)
            if k in instance.dates:
                if isinstance(val, list):
                    l = []
                    for v in val:
                        l.append(util.ymd(v))
                    val = l
                else:
                    val = util.ymd(val)
            elif k in instance.datetimes:
                if isinstance(val, list):
                    l = []
                    for v in val:
                        l.append(util.ymd_hmsf(v))
                    val = l
                else:
                    val = util.ymd_hmsf(val)
            # else:
            #     raise ValueError(f"do not know what to do about this: '{val}'")
            setattr(instance, k, val)
        instance.after_deserialize()
        instance = instance.unwrap()
        return instance
    elif isinstance(deserialized, dict):
        new_dict = {}
        for k, v in deserialized.items():
            if k == "k__" or k == "m_":
                continue
            val = ser_instanciate(v)
            new_dict[k] = val
        return new_dict
    elif isinstance(deserialized, list):
        l = []
        for v in deserialized:
            val = ser_instanciate(v)
            l.append(val)
        return l
    else:
        return deserialized


def __new_instance(k, m) -> Ser:
    mod = import_module(m)
    klass = getattr(mod, k)
    return klass()


def from_json(js):
    js = json.loads(js)
    instance = ser_instanciate(js)
    return instance


def to_json(instance, file: [str, Path] = None, complete=True, pretty_print=False, calcName=True):
    import json
    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, o):
            # print(f"debug '{o}: {type(o)}'")
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, type):
                return TW.wrap(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return NP.wrap(o)
            elif isinstance(o, Ser):
                d = {k: v for (k, v) in o.__dict__.items() if v is not None}
                for k in o.ignored:
                    if k in d:
                        d.pop(k)
                return d
            elif isinstance(o, date):
                s = o.__str__()
                return s
            elif isinstance(o, set):
                return list(o)
            elif isinstance(o, Path):
                return PW.wrap(o)
            return json.JSONEncoder.default(self, o)

    if pretty_print:
        js = json.dumps(instance, sort_keys=True, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    else:
        js = json.dumps(instance, sort_keys=True, ensure_ascii=False, cls=NumpyEncoder)
    if file is not None:
        write_text_file(file, js)
    return js

# if __name__ == '__main__':
#     f = "/mnt/sd1/Develop/priv/bepy88/config/zoo/t_bundesliga/run.0/state.json"
#     from keta.state import State
#
#     s: State = load_json(f)
#     print(s.latest_model_valid_until)
#     print(type(s.latest_model_valid_until))
