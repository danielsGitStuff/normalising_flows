from pathlib import Path

import numpy as np
from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov
from distributions.base import BaseMethods, enable_memory_growth
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.visual_random.VisualRandomExample import VisualRandomExample


class EvalExample4(VisualRandomExample):

    def __init__(self):
        self.input_dimensions: int = 5
        super().__init__('EvalExample4')

    def create_data_distribution(self) -> Distribution:
        rng = np.random.default_rng(45)
        sample_f = lambda: rng.normal(scale=2.0)
        cov = None
        enable_memory_growth()
        for seed in range(1905, 10000):
            Global.set_seed(seed)
            # sample_f = lambda: np.random.normal(scale=2.0)
            cov = BaseMethods.random_covariance_matrix(self.input_dimensions, sample_f=sample_f)
            m = tf.linalg.cholesky(cov)
            if not tf.reduce_any(tf.math.is_nan(m)):
                break
        print(f"seed {seed} works!")


        # cov = BaseMethods.random_covariance_matrix(n=self.input_dimensions, sample_f=sample_f)
        loc = rng.uniform(-3.0, 3.0, self.input_dimensions).astype(np.float32)
        print('data distribution')
        print('loc')
        print(loc)
        print('cov')
        print(cov)
        d = GaussianMultivariateFullCov(loc=loc, cov=cov)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=self.input_dimensions, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in
                [1, 3, 5]]

    def create_data_title(self) -> str:
        return '5D offset Gaussian'


if __name__ == '__main__':

    import tensorflow as tf
    enable_memory_growth()
    for seed in range(1905, 10000):
        Global.set_seed(seed)
        sample_f = lambda: np.random.normal(scale=2.0)
        cov = BaseMethods.random_covariance_matrix(5, sample_f=sample_f)
        m = tf.linalg.cholesky(cov)
        if not tf.reduce_any(tf.math.is_nan(m)):
            break
    print(f"seed {seed} works!")

    Global.set_seed(seed)
    EvalExample4().run()
