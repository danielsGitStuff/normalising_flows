from typing import Optional, List

import numpy as np
from keras.keras_parameterized import TestCase
from tensorflow_probability.python.distributions import Distribution as TD

from common.NotProvided import NotProvided
from distributions.Distribution import Distribution
from distributions.Uniform import Uniform
from distributions.base import TTensor, TTensorOpt, cast_to_ndarray


class UniformMultivariate(Distribution):
    def __init__(self, input_dim: int = NotProvided(), lows: List[float] = NotProvided(), highs: List[float] = NotProvided()):
        super().__init__(input_dim)
        self.lows: np.ndarray = self.cast_2_float_ndarray(lows)
        self.highs: np.ndarray = self.cast_2_float_ndarray(highs)
        self.uniforms: List[Uniform] = None

    def _create_base_distribution(self) -> Optional[TD]:
        self.uniforms: List[Uniform] = [Uniform(low=low, high=high) for low, high in zip(self.lows, self.highs)]
        return None

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        log_likelihoods = [u.log_likelihoods(xs[:, d]) for d, u in enumerate(self.uniforms)]
        ls = np.sum(log_likelihoods, axis=0)
        return self.cast_2_likelihood(input_tensor=xs, result=ls)

    def _sample(self, size: int = 1, cond: TTensorOpt = None) -> np.ndarray:
        samples = [u.sample(size) for u in self.uniforms]
        result = np.column_stack(samples)
        return cast_to_ndarray(result)

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        likelihoods = [u.likelihoods(xs[:, d]) for d, u in enumerate(self.uniforms)]
        ls = np.prod(likelihoods, axis=0)
        return self.cast_2_likelihood(input_tensor=xs, result=ls)


class UniformMultivariateTest(TestCase):
    def setUp(self):
        self.u = UniformMultivariate(input_dim=2, lows=[1.0, 2.0], highs=[2.5, 2.5])
        self.a: np.ndarray = np.array([[0.0, 2.0], [1.0, 2.0]])

    def test_a(self):
        ll = self.u.likelihoods(self.a)
        self.assertAllEqual(ll, np.array([[0.0], [4 / 3]], dtype=np.float32))

    def test_sample(self):
        samples = self.u.sample(3)
        print(samples)
