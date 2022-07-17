from maf.dim10.RandomExample10D import RandomExample10D
from maf.dim2.VisualRandomExample2D import VisualRandomExample2D
from typing import List, Optional

import numpy as np

from distributions.distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import enable_memory_growth, BaseMethods
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow


class EvalLargeD10(RandomExample10D):
    def __init__(self, name: str, layers: Optional[List[int]] = None, layers_repeat: int = 6, loc_range: float = 10.0, pool_size: int = 6):
        self.input_dimensions: int = 10
        self.no_of_gaussians: int = 7
        layers = layers or [3, 5, 7, 10, 20]
        super().__init__(name, layers=layers, layers_repeat=layers_repeat, loc_range=loc_range, pool_size=pool_size)
        self.epochs = 2000
        self.divergence_metric_every_epoch = 10
        # self.divergence_sample_size = 1024 * 400
        # self.no_val_samples = 1024 * 10
        # self.no_samples = 1024 * 100
        self.divergence_sample_size = 1024 * 300
        self.no_val_samples = 1024 * 12
        self.no_samples = 1024 * 800
        self.batch_size = 1024 * 4

    def create_data_distribution(self) -> Distribution:
        rng = np.random.default_rng(45)
        # enable_memory_growth()

        d = WeightedMultimodalMultivariate(input_dim=self.input_dimensions)
        for _ in range(self.no_of_gaussians):
            cov = BaseMethods.random_positive_semidefinite_matrix(self.input_dimensions, seed=int(np.random.random() * 1000000))
            weight = rng.random() + 3
            loc = rng.uniform(-self.loc_range, self.loc_range, self.input_dimensions).astype(np.float32)
            g = GaussianMultivariateFullCov(loc=loc, cov=cov)
            d.add_d(g, weight=weight)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=self.input_dimensions, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers
                in self.get_layers()]

    def create_data_title(self) -> str:
        return f'{self.no_of_gaussians}x{self.input_dimensions}D offset Gaussians, loc=[-{self.loc_range},{self.loc_range}]'
