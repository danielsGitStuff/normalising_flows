from typing import List, Optional

import numpy as np

from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import enable_memory_growth, BaseMethods
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim10.VisualRandomExample import VisualRandomExample


class EvalLargeD(VisualRandomExample):
    def __init__(self, name: str, layers: Optional[List[int]] = None):
        self.input_dimensions: int = 10
        self.loc_range: float = 15.0
        self.no_of_gaussians: int = 7
        super().__init__(name)
        self.epochs = 2000
        self.divergence_metric_every_epoch = 10
        # self.divergence_sample_size = 1024 * 400
        # self.no_val_samples = 1024 * 10
        # self.no_samples = 1024 * 100
        self.divergence_sample_size = 1024 * 200
        self.no_val_samples = 1024 * 4
        self.no_samples = 1024 * 100
        self.batch_size = 1024 * 2

    def create_data_distribution(self) -> Distribution:
        rng = np.random.default_rng(45)
        enable_memory_growth()

        d = WeightedMultimodalMultivariate(input_dim=self.input_dimensions)
        for _ in range(self.no_of_gaussians):
            cov = BaseMethods.random_positive_semidefinite_matrix(self.input_dimensions)
            weight = rng.random() + 3
            loc = rng.uniform(-self.loc_range, self.loc_range, self.input_dimensions).astype(np.float32)
            g = GaussianMultivariateFullCov(loc=loc, cov=cov)
            d.add_d(g, weight=weight)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=self.input_dimensions, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers
                in self.layers]

    def create_data_title(self) -> str:
        return f'{self.no_of_gaussians}x{self.input_dimensions}D offset Gaussians, loc=[-{self.loc_range},{self.loc_range}]'
