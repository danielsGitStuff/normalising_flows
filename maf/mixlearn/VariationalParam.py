from __future__ import annotations

from typing import List, Union, Optional

import numpy as np

from common.NotProvided import NotProvided

IntFloat = Union[int, float]


class VariationalParam:
    @staticmethod
    def fixed(name: str, value: IntFloat, int_type: bool = True) -> VariationalParam:
        p = VariationalParam(name=name, range_start=value, range_end=value, range_steps=0, int_type=int_type)
        dtype = np.int32 if int_type else np.float32
        p.range = np.array([value], dtype=dtype)
        return p

    @staticmethod
    def copy_of_column(name: str, source: str, int_type: bool = True) -> VariationalParam:
        p = VariationalParam(name=name, int_type=int_type)
        p.__scr_column = source
        return p

    def __init__(self, name: str = NotProvided(), range_start: IntFloat = NotProvided(), range_end: IntFloat = NotProvided(), range_steps: int = NotProvided(),
                 int_type: bool = True):
        self.name: str = NotProvided.value_if_not_provided(name, 'no name provided')
        self.range_start: IntFloat = range_start
        self.range_end: IntFloat = range_end
        self.range_steps: int = range_steps
        self.int_type: bool = int_type
        self.fixed: bool = False
        self.range: Optional[np.ndarray] = None

    def get_range(self) -> np.ndarray:
        if self.range is None:
            if self.int_type:
                self.range = np.ceil(np.linspace(self.range_start, self.range_end, self.range_steps, dtype=np.int32))
            self.range = np.linspace(self.range_start, self.range_end, self.range_steps, dtype=np.float32)
        return self.range
