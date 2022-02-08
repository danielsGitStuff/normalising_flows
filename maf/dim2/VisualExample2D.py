from typing import List

from abc import ABC

from maf.stuff.DivergenceExperiment import DivergenceExperiment


class VisualExample2D(DivergenceExperiment, ABC):
    def __init__(self, name: str, layers: List[int] = None, layers_repeat: int = 4):
        layers = layers or [1, 3, 5, 10]
        super().__init__(name, layers=layers, layers_repeat=layers_repeat)
        self.divergence_metric_every_epoch = 10

    def results_dir_name(self) -> str:
        return 'results_artificial_dim2'
