from typing import List, Optional

import numpy as np
from tensorflow_probability.python.distributions import Distribution as TD

from common.NotProvided import NotProvided
from distributions.distribution import Distribution
from distributions.base import TTensor, TTensorOpt, cast_to_ndarray


class MultimodalDistribution(Distribution):
    # todo merge with WeightedMultimodalMultivariate
    """Glues together multiple Distributions of the same dimension an makes them behave like one."""

    def _create_base_distribution(self) -> Optional[TD]:
        return None

    def __init__(self, input_dim: int = NotProvided(), distributions: List[Distribution] = NotProvided()):
        super().__init__(input_dim)
        self.distributions: List[Distribution] = distributions

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        log_likelihoods = [g.log_likelihoods(xs=xs, batch_size=batch_size) for g in self.distributions]
        log_likelihoods_sum = log_likelihoods[0]
        for ll in log_likelihoods[1:]:
            log_likelihoods_sum = np.logaddexp(log_likelihoods_sum, ll)
        result = log_likelihoods_sum - np.log(len(self.distributions))
        return self.cast_2_likelihood(input_tensor=xs, result=result)
        # return np.log(self.likelihoods(xs))

    def _sample(self, size: int = 1, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        self.check_condition(cond)
        vss = []
        distribution_idx = np.random.choice([i for i, _ in enumerate(self.distributions)], size)
        distribution_idx, counts = np.unique(distribution_idx, return_counts=True)
        for distribution_index, count in zip(distribution_idx, counts):
            distribution = self.distributions[distribution_index]
            vs = distribution.sample(size=count)
            vss.append(vs)
        vss = np.concatenate(vss)
        return cast_to_ndarray(vss)

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        likelihoods = [g.likelihoods(xs=xs, batch_size=batch_size) for g in self.distributions]
        likelihoods = np.stack(likelihoods, axis=1)
        result = np.sum(likelihoods, axis=1) / len(self.distributions)
        return self.cast_2_likelihood(input_tensor=xs, result=result)
