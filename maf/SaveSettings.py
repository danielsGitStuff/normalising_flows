from __future__ import annotations
from typing import List, Optional


class SaveSettings:
    def __init__(self, ranges_start: List[float] = (-13.0, -13.0), ranges_stop: List[float] = (13.0, 13.0), step_size: float = 0.1, prefix: Optional[str] = None,
                 print_3d: bool = False, show_3d: bool = False, suffix_by_param: Optional[str] = None, base_dir_ext: str = ""):
        assert not (suffix_by_param is not None and prefix is not None)  # just one!
        self.ranges_start: List[float] = ranges_start
        self.ranges_stop: List[float] = ranges_stop
        self.step_size: float = step_size
        self._prefix: Optional[str] = prefix
        self._print_3d: bool = print_3d
        self._show_3d: bool = show_3d
        self._suffix_by_param: Optional[str] = suffix_by_param
        self.base_dir_ext: str = base_dir_ext

    def get_prefix(self) -> str:
        if self._prefix is None and self._suffix_by_param is None:
            return ""
        elif self._prefix is not None:
            return f"{self._prefix}_"
        else:
            return f"{self._suffix_by_param}"

    def print_3d(self, value=True) -> SaveSettings:
        self._print_3d = value
        return self

    def show_3d(self, value=True) -> SaveSettings:
        self._show_3d = value
        return self

    def wants_3d_print(self) -> bool:
        return self._print_3d

    def wants_3d_shown(self) -> bool:
        return self._show_3d

    def get_base_dir(self):
        return f"ResultImages_{self.base_dir_ext}"
