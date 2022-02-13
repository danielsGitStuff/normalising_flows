from typing import List

from abc import ABC

from maf.stuff.DivergenceExperiment import DivergenceExperiment


class VisualExample1D(DivergenceExperiment, ABC):
    def __init__(self, name: str, layers: List[int] = None, layers_repeat: int = 3):
        layers = layers or [1, 3]
        super().__init__(name, layers=layers, layers_repeat=layers_repeat)
        self.divergence_metric_every_epoch = 1
        self.epochs = 20

    def results_dir_name(self) -> str:
        return 'results_artificial_dim1'
