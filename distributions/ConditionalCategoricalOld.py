from typing import List, Optional

import numpy as np

from common.NotProvided import NotProvided
from distributions.distribution import Distribution
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import TTensorOpt, TTensor


class ConditionalCategoricalOld(WeightedMultimodalMultivariate):
    """Condition Distributions on categories"""

    def __init__(self, input_dim: int = NotProvided(), categorical_dims: int = 1, conditional_dims: int = 1):
        assert conditional_dims >= categorical_dims
        self.categorical_dims: int = categorical_dims
        super().__init__(input_dim=input_dim, conditional_dims=conditional_dims)

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        if cond is None or NotProvided.is_not_provided(cond):
            raise RuntimeError("ConditionalCategorical is a conditional distribution. _log_likelihoods() was called without 'cond'.")
        results: List[np.ndarray] = []
        for index, x in zip(cond, xs):
            d: Distribution = self.get_distribution(int(index))
            results.append(d.log_likelihoods(x))
        results: np.ndarray = np.array(results, dtype=np.float32)
        return self.cast_2_likelihood(input_tensor=xs, result=results[:, 0, 0])

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        if cond is None or NotProvided.is_not_provided(cond):
            raise RuntimeError("ConditionalCategorical is a conditional distribution. _likelihoods() was called without 'cond'.")
        results: List[np.ndarray] = []
        for cs, x in zip(cond, xs):
            index = int(cs[:self.categorical_dims])
            c = cs[self.categorical_dims:]
            if c.shape == [0]:
                c = None
            d: Distribution = self.get_distribution(int(index))
            results.append(d.likelihoods(x, cond=c))
        results: np.ndarray = np.array(results, dtype=np.float32)
        return self.cast_2_likelihood(input_tensor=xs, result=results[:, 0, 0])

    def sample(self, size: int = 1, cond: TTensorOpt = None, batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        if cond is None:
            dist_indices: List[int] = list(range(len(self.distributions)))
            cond = np.random.choice(dist_indices, size, p=self.normalised_distribution_weights)
        samples = super(ConditionalCategoricalOld, self).sample(size=size, cond=cond, batch_size=batch_size, **kwargs)
        return np.column_stack([samples, cond])
