from __future__ import annotations

import sys

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalFullCovariance, MultivariateNormalTriL

from common.NotProvided import NotProvided
from distributions.Distribution import Distribution, TfpD
from distributions.base import TTensor, TTensorOpt
from typing import Optional
from tensorflow_probability.python.distributions.distribution import Distribution as DD


class GaussianMultivariateFullCov(Distribution):

    def __init__(self, loc: np.ndarray = NotProvided(), cov: np.ndarray = NotProvided()):
        input_dim = None
        if NotProvided.is_provided(cov):
            cov = self.cast_2_float_ndarray(cov)
            input_dim = cov.shape[0]
        if NotProvided.is_provided(loc):
            loc = self.cast_2_float_ndarray(loc)
        super().__init__(input_dim=input_dim, conditional_dims=0)
        self.cov: np.ndarray = cov
        self.loc: np.ndarray = loc

    def after_deserialize(self):
        self.input_dim = self.cov.shape[0]

    def _create_base_distribution(self) -> Optional[TfpD]:
        m = tf.linalg.cholesky(self.cov)
        if tf.reduce_any(tf.math.is_nan(m)):
            print('Cholesky decomposition failed for matrix:', file=sys.stderr)
            print(self.cov, file=sys.stderr)
            raise ValueError('stupid matrix. see log.')
        return MultivariateNormalTriL(scale_tril=m, loc=self.loc)
        # return MultivariateNormalTriL(scale_tril=self.cov, loc=self.loc)

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        return self.tfd_distribution.log_prob(xs)

    def _sample(self, size: int = 1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        samples = self.tfd_distribution.sample(sample_shape=(size,))
        return samples
        return self.cast_2_float_ndarray(samples)

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        return self.tfd_distribution.prob(xs)
