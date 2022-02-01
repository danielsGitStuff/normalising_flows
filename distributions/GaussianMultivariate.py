from typing import List, Union, Optional

import numpy as np
from tensorflow_probability.python.distributions import MultivariateNormalDiag

from common.NotProvided import NotProvided
from distributions.Distribution import Distribution, TfpD
from distributions.base import cast_to_ndarray, TTensor, TTensorOpt


class GaussianMultivariate(Distribution):
    def __init__(self, input_dim: int = NotProvided(), mus: List[float] = NotProvided(), cov: Union[List[float], np.ndarray] = NotProvided()):
        super().__init__(input_dim)
        self.mus: np.ndarray = self.cast_2_float_ndarray(mus)
        self.cov_matrix: np.ndarray = self.cast_2_float_ndarray(cov)

    def _create_base_distribution(self) -> Optional[TfpD]:
        assert len(self.mus) == len(self.cov_matrix) == self.input_dim
        return MultivariateNormalDiag(loc=self.mus, scale_diag=np.sqrt(self.cov_matrix).astype(np.float32), )

    def _sample(self, size: int = 1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        samples = self.tfd_distribution.sample(size)
        return cast_to_ndarray(samples)

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        xs, _ = self.extract_xs_cond(xs, cond)
        return self.cast_2_likelihood(input_tensor=xs, result=self.tfd_distribution.log_prob(xs))

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        xs, _ = self.extract_xs_cond(xs)
        return self.cast_2_likelihood(input_tensor=xs, result=self.tfd_distribution.prob(xs))
