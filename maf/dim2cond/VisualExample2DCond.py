from abc import ABC

from typing import List

from maf.stuff.ConditionalConvergenceExperiment import ConditionalDivergenceExperiment
from maf.stuff.DivergenceExperiment import DivergenceExperiment


class VisualExample2DCond(ConditionalDivergenceExperiment, ABC):
    def __init__(self, name: str, layers: List[int] = None, layers_repeat: int = 4, pool_size: int = 6):
        layers = layers or [1, 3, 5, 10]
        super().__init__(name, layers=layers, layers_repeat=layers_repeat, pool_size=pool_size)
        self.divergence_metric_every_epoch = 10

    def results_dir_name(self) -> str:
        return 'results_artificial_dim2_cond'
