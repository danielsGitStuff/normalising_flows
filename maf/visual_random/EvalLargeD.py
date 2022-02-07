from typing import List

import numpy as np

from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import enable_memory_growth, BaseMethods
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.visual_random.VisualRandomExample import VisualRandomExample


class EvalLargeD(VisualRandomExample):
    def __init__(self, name: str):
        self.input_dimensions: int = 10
        self.loc_range: float = 10.0
        self.no_of_gaussians: int = 7
        super().__init__(name)
        self.epochs = 250
        self.divergence_metric_every_epoch = 10
        self.divergence_sample_size = 1024 * 400
        self.no_val_samples = 1024 * 10
        self.no_samples = 1024 * 100

    def create_data_distribution(self) -> Distribution:
        rng = np.random.default_rng(45)
        enable_memory_growth()

        def sample_f() -> float:
            s = np.abs(rng.normal(scale=.2)) + .2
            s = s * rng.choice([1, -1])
            return s

        d = WeightedMultimodalMultivariate(input_dim=self.input_dimensions)
        for _ in range(self.no_of_gaussians):
            cov = BaseMethods.random_positive_semidefinite_matrix(self.input_dimensions, sample_f=sample_f)
            weight = rng.random() + 3
            loc = rng.uniform(-self.loc_range, self.loc_range, self.input_dimensions).astype(np.float32)
            g = GaussianMultivariateFullCov(loc=loc, cov=cov)
            d.add_d(g, weight=weight)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=self.input_dimensions, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in
                [5, 5, 10, 10]]

    def create_data_title(self) -> str:
        return f'{self.no_of_gaussians}x{self.input_dimensions}D offset Gaussians, loc=[-{self.loc_range},{self.loc_range}]'
