from typing import List

from abc import ABC

from maf.stuff.DivergenceExperiment import DivergenceExperiment


class VisualRandomExample(DivergenceExperiment, ABC):
    def __init__(self, name: str, layers: List[int] = None, layers_repeat: int = 1):
        super().__init__(name, layers=layers, layers_repeat=layers_repeat)

    def results_dir_name(self) -> str:
        return 'results_artificial_dim10'
