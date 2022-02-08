from pathlib import Path

import numpy as np
import tensorflow as tf
from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import BaseMethods, enable_memory_growth
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim10.VisualRandomExample import VisualRandomExample


class EvalExample3(VisualRandomExample):

    def __init__(self):
        self.input_dimensions: int = 2
        super().__init__('EvalExample3')
        self.mesh_count = 500
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.epochs = 2000
        self.patiences = [100, 100, 100]
        self.no_samples = 30000
        self.no_val_samples = 3000

    def create_data_distribution(self) -> Distribution:
        d = WeightedMultimodalMultivariate(input_dim=self.input_dimensions)

        no_of_distributions = 7
        rng = np.random.default_rng(42)

        def sample_f() -> float:
            s = np.abs(rng.normal(scale=.2)) + .2
            s = s * rng.choice([1, -1])
            return s

        # sample_f = lambda: rng.normal(scale=0.2)+ .8

        for i in range(no_of_distributions):
            weight = rng.random() + 3
            loc = rng.uniform(-7.0, 7.0, self.input_dimensions)
            cov = BaseMethods.random_positive_semidefinite_matrix(n=self.input_dimensions, sample_f=sample_f)
            cov = cov / cov.max()
            # cov[0][0] = 1.0
            # cov[1][1] = 1.0
            print(cov)
            g = GaussianMultivariateFullCov(loc=loc, cov=cov)
            d.add_d(g, weight=weight)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=self.input_dimensions, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True,
                                         batch_norm=False) for layers
                in [5]]

    def create_data_title(self) -> str:
        return f"7 Gaussians"


if __name__ == '__main__':
    enable_memory_growth()
    EvalExample3().run()
