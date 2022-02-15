from typing import List

from abc import ABC

from maf.stuff.DivergenceExperiment import DivergenceExperiment


class RandomExample10D(DivergenceExperiment, ABC):
    def __init__(self, name: str, layers: List[int] = None, layers_repeat: int = 4, loc_range: float = 15.0):
        self.loc_range: float = loc_range
        super().__init__(name, layers=layers, layers_repeat=layers_repeat)

    def results_dir_name(self) -> str:
        return 'results_artificial_dim10'
