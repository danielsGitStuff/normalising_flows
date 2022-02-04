from abc import ABC

from maf.stuff.DivergenceExperiment import DivergenceExperiment


class VisualRandomExample(DivergenceExperiment, ABC):
    def __init__(self, name: str):
        super().__init__(name)

    def results_dir_name(self) -> str:
        return 'results_artificial_random'