import sys
from pathlib import Path

import numpy as np

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import BaseMethods
from distributions.kl.JS import JensenShannonDivergence
from distributions.kl.KL import KullbackLeiblerDivergence
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.DivergenceExperiment import DivergenceExperiment
from maf.stuff.Foursome2DExample import Foursome2DMafExperiment
from maf.stuff.MafExperiment import MafExperiment
from typing import List
import tensorflow as tf


class EvalExample3(DivergenceExperiment):

    def __init__(self):
        self.input_dimensions: int = 2
        super().__init__('Eval3')
        self.mesh_count = 500
        self.divergence_sample_size = 10000
        self.no_samples = 24000
        self.no_val_samples = 1500
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.epochs = 2000
        self.use_early_stop = True
        self.patiences = [50, 50, 50]

    def create_data_distribution(self) -> Distribution:
        sample_f = lambda: np.random.normal(scale=2.0)
        cov = BaseMethods.random_covariance_matrix(n=self.input_dimensions, sample_f=sample_f)
        # cov = [[1.0, 0.5], [0.5, 2.0]]
        # cov = np.array(cov, dtype=np.float32)
        loc = np.array([0] * self.input_dimensions, dtype=np.float32)
        d = GaussianMultivariateFullCov(loc=loc, cov=cov)
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
        return [MaskedAutoregressiveFlow(input_dim=self.input_dimensions, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers
                in [3, 5]]

    def create_data_title(self) -> str:
        return f"7 Gaussians"

    def _run(self):
        self._print_datadistribution()
        super(EvalExample3, self)._run()

    # def _print_datadistribution(self):
    #     if self.data_distribution.input_dim == 2:
    #         print('printing dataset')
    #         self.hm(self.data_distribution)
    #
    #         self.print_denses(name=f"{self.name}_data")
    #         # sys.exit(7)


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    EvalExample3().run()
