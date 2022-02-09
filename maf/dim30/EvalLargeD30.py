from maf.dim10.EvalLargeD10 import EvalLargeD10
from typing import List, Optional

import numpy as np

from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import enable_memory_growth, BaseMethods
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow


class EvalLargeD30(EvalLargeD10):
    def __init__(self, name: str, layers: Optional[List[int]] = None, layers_repeat: int = 3, loc_range: float = 15.0):
        self.input_dimensions: int = 20
        self.loc_range: float = 15.0
        self.no_of_gaussians: int = 9
        super().__init__(name, layers=layers, layers_repeat=layers_repeat, loc_range=loc_range)
        self.epochs = 2000
        self.divergence_metric_every_epoch = 10
        self.divergence_sample_size = 1024 * 300
        self.no_val_samples = 1024 * 8
        self.no_samples = 1024 * 300
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

    def create_data_title(self) -> str:
        return f'{self.no_of_gaussians}x{self.input_dimensions}D offset Gaussians, loc=[-{self.loc_range},{self.loc_range}]'

    def results_dir_name(self) -> str:
        return 'results_artificial_dim30'
