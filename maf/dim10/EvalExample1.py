from distributions.base import BaseMethods
from pathlib import Path

import numpy as np
from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim10.VisualRandomExample import VisualRandomExample


class EvalExample1(VisualRandomExample):

    def __init__(self):
        super().__init__('EvalExample1', layers=[1, 3], layers_repeat=3)

    def create_data_distribution(self) -> Distribution:
        cov = BaseMethods.random_positive_semidefinite_matrix(10, seed=77)
        loc = np.zeros(10)
        d = GaussianMultivariateFullCov(loc=loc, cov=cov)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=10, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in self.get_layers()]

    def create_data_title(self) -> str:
        return 'X ~ N([0], [10xRandom])'


if __name__ == '__main__':
    Global.set_seed(42)
    EvalExample1().run()
