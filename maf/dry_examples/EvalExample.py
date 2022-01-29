import sys
from pathlib import Path

import numpy as np

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.kl.JS import JensenShannonDivergence
from distributions.kl.KL import KullbackLeiblerDivergence
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.Foursome2DExample import Foursome2DMafExperiment
from maf.stuff.MafExperiment import MafExperiment
from typing import List


class EvalExample(Foursome2DMafExperiment):

    def __init__(self):
        super().__init__('EvalExample')
        self.mesh_count = 100
        self.divergence_half_width = 4.0
        self.divergence_step_size = 0.5

    def create_data_distribution(self) -> Distribution:
        cov = [[1.0, 0.5], [0.5, 2.0]]
        cov = np.array(cov, dtype=np.float32)
        loc = np.array([0, 0], dtype=np.float32)
        d = GaussianMultivariateFullCov(loc=loc, cov=cov)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in [1, 3]]

    def create_data_title(self) -> str:
        return 'testi!!!'

    def _run(self):
        self._print_datadistribution()
        super(EvalExample, self)._run()

    def _print_datadistribution(self):
        if self.data_distribution.input_dim == 2:
            print('printing dataset')
            self.hm(self.data_distribution)

            self.print_denses(name=f"{self.name}_data")
            # sys.exit(7)


if __name__ == '__main__':
    Global.set_seed(42)
    Global.set_global('results_dir', Path('results_artificial'))
    EvalExample().run()
