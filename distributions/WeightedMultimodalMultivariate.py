from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np

from common.NotProvided import NotProvided
from distributions.Distribution import Distribution, TfpD
from distributions.UniformMultivariate import UniformMultivariate
from distributions.base import TTensor, TTensorOpt, BaseMethods
import tensorflow as tf


class WeightedMultimodalMultivariate(Distribution):
    # todo merge with MultimodalDistribution
    def __init__(self, input_dim: int = NotProvided(), conditional_dims: int = 0):
        super().__init__(input_dim=input_dim, conditional_dims=conditional_dims)
        self._distributions: List[Distribution] = []
        self._distribution_weights: List[float] = []
        self.weight_sum: float = 0.0
        self._normalised_distribution_weights: Optional[List[float]] = None
        self.ignored.add('_normalised_distribution_weights')

    def normalised_distribution_weights(self) -> List[float]:
        if self._normalised_distribution_weights is None:
            self._normalised_distribution_weights = [weight / self.weight_sum for d, weight in zip(self._distributions, self._distribution_weights)]
        return self._normalised_distribution_weights

    def _create_base_distribution(self) -> Optional[TfpD]:
        return None

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        weights = np.log(self.normalised_distribution_weights())
        ll_sum = None
        for weight, d in zip(weights, self._distributions):
            ls = d.log_likelihoods(xs=xs, cond=cond) + weight
            if ll_sum is None:
                ll_sum = ls
            else:
                ll_sum = np.logaddexp(ll_sum, ls)
        return self.cast_2_likelihood(input_tensor=xs, result=ll_sum)

    def _sample(self, size: int = 1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        weights = self.normalised_distribution_weights()
        vss = []
        distribution_idx = np.random.choice([i for i, _ in enumerate(self._distributions)], size, p=weights)
        distribution_idx, counts = np.unique(distribution_idx, return_counts=True)
        for distribution_index, count in zip(distribution_idx, counts):
            distribution = self._distributions[distribution_index]
            vs = distribution.sample(size=count)
            vss.append(vs)
        vss = np.concatenate(vss)
        return self.cast_2_float_ndarray(vss)

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        weights = self.normalised_distribution_weights()
        ll_sum = None
        for weight, d in zip(weights, self._distributions):
            ls = d.likelihoods(xs=xs, cond=cond) * weight
            if ll_sum is None:
                ll_sum = ls
            else:
                ll_sum += ls
        ll_sum = ll_sum  # / len(self._distributions)
        return self.cast_2_likelihood(input_tensor=xs, result=ll_sum)

    def add_d(self, d: Distribution, weight: float) -> WeightedMultimodalMultivariate:
        self._distributions.append(d)
        self._distribution_weights.append(weight)
        self.weight_sum += weight
        self._normalised_distribution_weights = None
        return self


if __name__ == '__main__':
    DIM: int = 2
    w = WeightedMultimodalMultivariate(input_dim=2)
    for i in range(4):
        src_offsets = np.random.random(DIM) * 7
        src_lows = np.random.random(DIM)
        src_highs = np.random.random(DIM)

        lows: List[float] = []
        highs: List[float] = []
        for o, l, h in zip(src_offsets, src_lows, src_highs):
            low = min(l, h)
            high = max(l, h)
            low += o
            high += o
            lows.append(low)
            highs.append(high)
        u = UniformMultivariate(input_dim=DIM, lows=lows, highs=highs)
