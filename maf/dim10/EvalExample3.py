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
        self.set_minmax_square(10.0)
        self.patiences = [100, 100, 100]
        self.layers = [1, 5, 10]

    def create_data_distribution(self) -> Distribution:
        d = WeightedMultimodalMultivariate(input_dim=self.input_dimensions)

        no_of_distributions = 7
        covs: List[np.ndarray] = []
        seed = -1
        while len(covs) < no_of_distributions:
            seed += 1
            Global.set_seed(seed)
            sample_f = lambda: np.random.normal(scale=2.0)
            cov = BaseMethods.random_covariance_matrix(self.input_dimensions, sample_f=sample_f)
            m = tf.linalg.cholesky(cov)
            if not tf.reduce_any(tf.math.is_nan(m)):
                covs.append(cov)
                print(f"seed {seed} works!")

        for cov in covs:
            weight = np.random.random() + 3
            loc = np.random.uniform(-7.0, 7.0, self.input_dimensions)
            g = GaussianMultivariateFullCov(loc=loc, cov=cov)
            d.add_d(g, weight=weight)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=self.input_dimensions, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True,
                                         batch_norm=False) for layers
                in self.get_layers()]

    def create_data_title(self) -> str:
        return f"7 Gaussians"


if __name__ == '__main__':
    enable_memory_growth()
    EvalExample3().run()
