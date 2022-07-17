from common.argparser import ArgParser
from maf.dim2.VisualRandomExample2D import VisualRandomExample2D
from pathlib import Path

import numpy as np
import tensorflow as tf
from typing import List

from common.globals import Global
from distributions.distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import BaseMethods, enable_memory_growth
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow


class NF2D_RandomA(VisualRandomExample2D):

    def __init__(self):
        self.input_dimensions: int = 2
        super().__init__('NF2D_RandomA', layers=[1, 3, 5, 10])
        self.mesh_count = 500
        self.set_minmax_square(10.0)

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
            try:
                np.linalg.cholesky(cov)
                covs.append(cov)
            except np.linalg.LinAlgError:
                print('np.linalg.error')

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
    ArgParser.parse()
    NF2D_RandomA().run()
