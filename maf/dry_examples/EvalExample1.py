from pathlib import Path

import numpy as np
from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.Foursome2DExample import Foursome2DMafExperiment


class EvalExample1(Foursome2DMafExperiment):

    def __init__(self):
        super().__init__('EvalExample1')
        self.mesh_count = 100

    def create_data_distribution(self) -> Distribution:
        cov = [[1.0, 0.5], [0.5, 2.0]]
        cov = np.array(cov, dtype=np.float32)
        loc = np.array([0, 0], dtype=np.float32)
        d = GaussianMultivariateFullCov(loc=loc, cov=cov)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in [1, 5,10]]

    def create_data_title(self) -> str:
        return 'X ~ N([0, 0], [[1.0, 0.5], [0.5, 2.0]])'

if __name__ == '__main__':
    Global.set_seed(42)
    Global.set_global('results_dir', Path('results_artificial'))
    EvalExample1().run()
