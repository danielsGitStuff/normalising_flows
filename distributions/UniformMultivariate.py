from typing import Optional, List

import numpy as np
from tensorflow_probability.python.distributions import Distribution as TD

from common.NotProvided import NotProvided
from distributions.Distribution import Distribution
from distributions.base import TTensor, TTensorOpt, cast_to_ndarray
from tensorflow_probability.python.distributions import Uniform as UU
import tensorflow as tf


class UniformMultivariate(Distribution):
    def __init__(self, input_dim: int = NotProvided(), lows: List[float] = NotProvided(), highs: List[float] = NotProvided()):
        super().__init__(input_dim)
        self.lows: np.ndarray = self.cast_2_float_ndarray(lows)
        self.highs: np.ndarray = self.cast_2_float_ndarray(highs)

    def _create_base_distribution(self) -> Optional[TD]:
        return UU(self.lows, self.highs)

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        ls = self.tfd_distribution.log_prob(xs)
        ls = tf.reduce_sum(ls, axis=1)
        return self.cast_2_likelihood(input_tensor=xs, result=ls)

    def _sample(self, size: int = 1, cond: TTensorOpt = None) -> np.ndarray:
        result = self.tfd_distribution.sample(sample_shape=(size,))
        return cast_to_ndarray(result)

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        ls = self.tfd_distribution.prob(xs)
        ls = tf.math.reduce_prod(ls, axis=1)
        return self.cast_2_likelihood(input_tensor=xs, result=ls)
