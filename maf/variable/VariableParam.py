from __future__ import annotations

import math
from typing import Optional, Callable, List, Type, TypeVar, Union, Any, Tuple

import numpy as np
from pandas import Series

from maf.variable.DependencyChecker import Dependency

Converter = Union[int, float, bool, Callable[[Any], Any]]


class Param:
    def __init__(self, name: str, converter: Converter = float, is_var: bool = False):
        """:converter function that casts the value as you wish"""
        self.name: str = name
        self.converter: Converter = converter
        self.is_var: bool = is_var

    def get_range(self) -> np.ndarray:
        raise NotImplementedError()

    def cast(self, value: Any) -> Any:
        return self.converter(value)

    def to_dependency(self) -> Dependency:
        return Dependency(name=self.name)


class FixedParam(Param):
    def __init__(self, name: str, value: float, converter: Converter = float, is_var: bool = False):
        super().__init__(name, is_var=is_var, converter=converter)
        self.value: float = float(value)

    def get_range(self) -> np.ndarray:
        return np.array([self.value], dtype=np.float32)


class FixedIntParam(FixedParam):
    def __init__(self, name: str, value: int, converter: Converter = float, is_var: bool = False):
        super().__init__(name, is_var=is_var, converter=converter, value=value)


class MetricParam(Param):
    def __init__(self, name: str, converter: Converter = float):
        super().__init__(name, converter=converter)


class MetricIntParam(MetricParam):
    def __init__(self, name: str):
        super().__init__(name, converter=int)


class VariableParam(Param):
    def __init__(self, name: str, range_start: float, range_end: float, range_steps: int, is_var: bool = False):
        super().__init__(name, is_var=is_var)
        self.range_start: float = range_start
        self.range_end: float = range_end
        self.range_steps: int = range_steps

    def get_range(self) -> np.ndarray:
        return np.linspace(self.range_start, self.range_end, self.range_steps, dtype=np.float32)


class VariableParamInt(VariableParam):
    def __init__(self, name: str, range_start: int, range_end: int, range_steps: int, is_var: bool = False):
        super().__init__(name, range_start, range_end, range_steps, is_var=is_var)

    def get_range(self) -> np.ndarray:
        return np.floor(np.linspace(self.range_start, self.range_end, self.range_steps, dtype=np.int32))


class LambdaParam(Param):
    def __init__(self, name: str, source_params: [str, List[str]], f: Callable[[], float], is_var: bool = False, converter: Converter = float):
        super().__init__(name, is_var=is_var, converter=converter)
        if isinstance(source_params, str):
            source_params = list([source_params])
        self.source_params: List[str] = source_params
        self.f: Callable[[...], float] = f

    def to_dependency(self) -> Dependency:
        return Dependency(name=self.name, src=self.source_params)


class LambdaIntParam(LambdaParam):
    def __init__(self, name: str, source_params: [str, List[str]], f: Callable[[], int], is_var: bool = False):
        super().__init__(name, is_var=is_var, converter=int, source_params=source_params, f=f)

    def to_dependency(self) -> Dependency:
        return Dependency(name=self.name, src=self.source_params)


class CopyFromParam(LambdaParam):
    def __init__(self, name: str, source_param: str):
        super().__init__(name, source_params=[source_param], f=None)

        def f(series: Series):
            p = series[source_param]
            return p

        self.f = f


class LambdaParam2(Param):
    def __init__(self, name: str, source_params: [str, List[str]], f: Callable[[], float], is_var: bool = False):
        super().__init__(name, is_var=is_var)
        if isinstance(source_params, str):
            source_params = list([source_params])
        self.source_params: List[str] = source_params
        self.f: Callable[[], float] = f


class LambdaParams:

    @staticmethod
    def clf_t_g_size_from_clfsize_synthratio(name: str = 'clf_t_g_size', val_size: Optional[int] = None, val_split: Optional[float] = None, ):
        """naming scheme:
        'clf_t_g_size_from_clfsize_synthratio'
        clf: this is fed into the classifier process
        [t,v]: this returns the amount of training or validation samples ...
        [g,s]: that are either genuine or synthetic ...
        from: obviously 'from' the source columns coming after this
        """

        def f(series: Series) -> float:
            clfsize: float = series['clfsize']
            synthratio: float = series['synthratio']
            val_take = val_size
            if val_split is not None:
                val_take = math.floor(clfsize * val_split)
            t_size = math.floor(clfsize - val_take)
            clf_t_s_size: int = math.floor(t_size * synthratio)
            return t_size - clf_t_s_size

        lp = LambdaParam(name, source_params=['clfsize', 'synthratio'], f=f)
        return lp

    @staticmethod
    def clf_t_s_size_from_clfsize_synthratio(name: str = 'clf_t_s_size', val_size: Optional[int] = None, val_split: Optional[float] = None, ):
        def f(series: Series) -> float:
            clfsize: float = series['clfsize']
            synthratio: float = series['synthratio']
            val_take = val_size
            if val_split is not None:
                val_take = math.floor(clfsize * val_split)
            t_size = math.floor(clfsize - val_take)
            clf_t_s_size: int = math.floor(t_size * synthratio)
            return clf_t_s_size

        lp = LambdaParam(name, source_params=['clfsize', 'synthratio'], f=f)
        return lp

    @staticmethod
    def clf_v_g_size_from_clfsize_synthratio(name: str = 'clf_v_g_size', val_size: Optional[int] = None, val_split: Optional[float] = None, ):
        def f(series: Series) -> float:
            clfsize: float = series['clfsize']
            synthratio: float = series['synthratio']
            val_take = val_size
            if val_split is not None:
                val_take = math.floor(clfsize * val_split)
            clf_v_s_size: int = math.floor(val_take * synthratio)
            return val_take - clf_v_s_size

        lp = LambdaParam(name, source_params=['clfsize', 'synthratio'], f=f)
        return lp

    @staticmethod
    def clf_v_s_size_from_clfsize_synthratio(name: str = 'clf_v_s_size', val_size: Optional[int] = None, val_split: Optional[float] = None, ):
        def f(series: Series) -> float:
            clfsize: float = series['clfsize']
            synthratio: float = series['synthratio']
            val_take = val_size
            if val_split is not None:
                val_take = math.floor(clfsize * val_split)
            clf_v_s_size: int = math.floor(val_take * synthratio)
            return clf_v_s_size

        lp = LambdaParam(name, source_params=['clfsize', 'synthratio'], f=f)
        return lp

    @staticmethod
    def clf_v_g_size_from_clf_t_g_size_clfsize(name: str = 'clf_v_g_size', val_size: Optional[int] = None, val_split: Optional[float] = None) -> LambdaParam:
        def f(series: Series) -> float:
            clfsize: float = series['clfsize']
            clf_t_g_size: float = series['clf_t_g_size']
            clf_t_s_size: float = series['clf_t_s_size']
            val_take = val_size
            if val_split is not None:
                val_take = math.floor(clfsize * val_split)
            return clfsize - clf_t_g_size - val_take - clf_t_s_size

        lp = LambdaParam(name, source_params=['clf_t_g_size', 'clf_t_s_size', 'clfsize'], f=f)
        return lp

    @staticmethod
    def clf_t_s_size_from_clf_t_g_size_clfsize(name: str = 'clf_t_s_size', val_size: Optional[int] = None, val_split: Optional[float] = None) -> LambdaParam:
        def f(series: Series) -> float:
            clfsize: float = series['clfsize']
            clf_t_g_size: float = series['clf_t_g_size']
            val_take = val_size
            if val_split is not None:
                val_take = math.floor(clfsize * val_split)
            return clfsize - clf_t_g_size - val_take

        lp = LambdaParam(name, source_params=['clf_t_g_size', 'clfsize'], f=f, is_var=True)
        return lp

    @staticmethod
    def clf_v_g_size_from_clf_t_g_size_clf_t_s_size(name: str = 'clf_v_g_size', val_size: Optional[int] = None) -> LambdaParam:
        def f(series: Series) -> float:
            clf_t_s_size: float = series['clf_t_s_size']
            clf_t_g_size: float = series['clf_t_g_size']
            if clf_t_s_size == 0:
                return val_size
            take = clf_t_g_size / (clf_t_g_size + clf_t_s_size) * val_size
            return math.floor(take)

        lp = LambdaParam(name, source_params=['clf_t_g_size', 'clf_t_s_size'], f=f)
        return lp

    @staticmethod
    def clf_v_s_size_from_clf_v_g_size(name: str = 'clf_v_s_size', val_size: Optional[int] = None, val_split: Optional[float] = None) -> LambdaParam:
        def f(series: Series) -> float:
            clf_v_g_size: float = series['clf_v_g_size']
            return math.floor(val_size - clf_v_g_size)

        lp = LambdaParam(name, source_params=['clf_v_g_size'], f=f)
        return lp

    @staticmethod
    def tsize_from_dsize(val_size: Optional[int] = None, val_split: Optional[float] = None, name: str = 'tsize') -> LambdaParam:
        def f(series: Series) -> float:
            dsize = series['dsize']
            if val_split is not None:
                vsize = math.floor(dsize * val_split)
                return dsize - vsize
            return dsize - val_size

        lp = LambdaParam(name, source_params='dsize', f=f)
        return lp

    @staticmethod
    def vsize_from_dsize(val_size: Optional[int] = None, val_split: Optional[float] = None, name: str = 'vsize') -> LambdaParam:
        def f(series: Series) -> float:
            dsize = series['dsize']
            if val_split is not None:
                vsize = math.floor(dsize * val_split)
                return vsize
            return val_size

        lp = LambdaParam(name, source_params='dsize', f=f)
        return lp
