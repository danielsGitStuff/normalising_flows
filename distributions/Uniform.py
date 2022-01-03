from __future__ import annotations

from typing import List, Optional

import numpy as np
from tensorflow_datasets.testing import TestCase
from tensorflow_probability.python.distributions import Distribution as TD

from common.NotProvided import NotProvided
from distributions.Distribution import Distribution
from distributions.base import TTensor, TTensorOpt, cast_to_ndarray
from tensorflow_probability.python.distributions import Uniform as U


class Uniform(Distribution):

    def __init__(self, low: float = NotProvided(), high: float = NotProvided()):
        super().__init__(1)
        self.low: float = low
        self.high: float = high
        self.distance: float = self.low - self.high

    def _create_base_distribution(self) -> Optional[TD]:
        return U(low=self.low, high=self.high)

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        return self.cast_2_likelihood(result=self.tfd_distribution.log_prob(xs), input_tensor=xs)

    def _sample(self, size: int = 1, cond: TTensorOpt = None) -> np.ndarray:
        return cast_to_ndarray(self.tfd_distribution.sample(size))

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        ll = self.tfd_distribution.prob(xs)
        return self.cast_2_likelihood(result=ll, input_tensor=xs)


class UniformTest(TestCase):
    def setUp(self):
        self.u = Uniform(low=1.0, high=2.5)
        self.a: np.ndarray = np.array([[0.1], [1.0], [1.1], [2.0]], dtype=np.float32)

    def test_ll(self):
        ll = self.u.likelihoods(self.a)
        self.assertAllEqual(ll, np.array([[0.0], [2 / 3], [2 / 3], [2 / 3]], dtype=np.float32))
