from distributions.base import BaseMethods
from maf.dim10.RandomExample10D import RandomExample10D
from maf.dim2.VisualRandomExample2D import VisualRandomExample2D
from pathlib import Path

import numpy as np
from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow


class Dim10aCenteredMVG(RandomExample10D):

    def __init__(self):
        super().__init__('Dim10aCenteredMVG', layers=[1, 2], layers_repeat=2)
        self.no_samples = 1024 * 100
        self.no_val_samples = 1024 * 10

    def create_data_distribution(self) -> Distribution:
        cov = BaseMethods.random_positive_semidefinite_matrix(10, seed=77)
        loc = np.zeros(10)
        d = GaussianMultivariateFullCov(loc=loc, cov=cov)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=10, layers=layers, activation="relu", use_tanh_made=True, hidden_shape=[200, 200], norm_layer=True) for layers in
                self.get_layers()]

    def create_data_title(self) -> str:
        return 'X ~ N([0], [10xRandom])'


if __name__ == '__main__':
    Global.set_seed(42)
    Dim10aCenteredMVG().run()
