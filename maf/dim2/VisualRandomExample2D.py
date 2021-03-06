from typing import List

from abc import ABC

from maf.stuff.DivergenceExperiment import DivergenceExperiment


class VisualRandomExample2D(DivergenceExperiment, ABC):
    def __init__(self, name: str, layers: List[int] = None, layers_repeat: int = 4, loc_range: float = 15.0, pool_size: int = 6):
        self.loc_range: float = loc_range
        super().__init__(name, layers=layers, layers_repeat=layers_repeat, pool_size=pool_size)

    def results_dir_name(self) -> str:
        return 'results_artificial'
