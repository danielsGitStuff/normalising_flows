from sklearn.datasets import make_spd_matrix

from common.argparser import ArgParser
from maf.dim2.VisualRandomExample2D import VisualRandomExample2D
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


class NF2D_RandomB(VisualRandomExample2D):

    def __init__(self):
        self.input_dimensions: int = 2
        super().__init__('NF2D_RandomB', layers=[1, 5, 10])
        self.mesh_count = 500
        self.set_minmax_square(15.0)
        self.layers_repeat = 1
        self.vmax = None

    def create_data_distribution(self) -> Distribution:
        d = WeightedMultimodalMultivariate(input_dim=self.input_dimensions)

        no_of_distributions = 8
        rng = np.random.default_rng(42 + 54)
        random_state = np.random.RandomState(42 + 54)

        for i in range(no_of_distributions):
            weight = rng.random() + 3
            loc = rng.uniform(-7.0, 7.0, self.input_dimensions)
            # cov = BaseMethods.random_positive_semidefinite_matrix(n=self.input_dimensions, seed=58)
            cov = BaseMethods.random_positive_semidefinite_matrix(self.input_dimensions, seed=int(rng.random() * 100000))
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
    ArgParser.parse()
    Global.set_global('results_dir', Path('results_dev'))
    NF2D_RandomB().run()
