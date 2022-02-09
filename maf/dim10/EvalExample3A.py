from sklearn.datasets import make_spd_matrix

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


class EvalExample3A(VisualRandomExample):

    def __init__(self):
        self.input_dimensions: int = 2
        super().__init__('EvalExample3A', layers=[1, 5, 10])
        self.mesh_count = 500
        self.set_minmax_square(15.0)
        self.layers_repeat = 1
        self.patiences = [100, 100, 100]
        self.vmax = None

    def create_data_distribution(self) -> Distribution:
        d = WeightedMultimodalMultivariate(input_dim=self.input_dimensions)

        no_of_distributions = 7
        rng = np.random.default_rng(42 + 54)

        for i in range(no_of_distributions):
            weight = rng.random() + 3
            loc = rng.uniform(-5.0, 5.0, self.input_dimensions)
            cov = BaseMethods.random_positive_semidefinite_matrix(n=self.input_dimensions, seed=58)
            print(cov)
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
    EvalExample3A().run()
