from __future__ import annotations

import math

import numpy as np

from distributions.Distribution import Distribution
from distributions.kl.KL import KullbackLeiblerDivergence
from distributions.kl.KLSampler import KLSampler

import tensorflow as tf
import keras.backend as K


class JensenShannonDivergence(KullbackLeiblerDivergence):
    def __init__(self, p: Distribution, q: Distribution, half_width: float, step_size: float, batch_size: int = 100000):
        super().__init__(p, q, half_width, step_size, batch_size)

    def calculate(self) -> float:
        sampler = KLSampler(dims=self.dims, half_width=self.half_width, step_size=self.step_size, batch_size=self.batch_size)
        print(f"will calculate JS divergence in {len(sampler.batch_sizes)} batches")
        js_sum: float = 0.0
        # log_epsilon = tf.cast(tf.math.log(K.epsilon()), dtype=tf.float32)
        epsilon = tf.cast(K.epsilon(), dtype=tf.float32)

        def kl(p: np.ndarray, q: np.ndarray) -> float:
            k = p * np.log2(p / q)
            return np.sum(k)

        for batch in sampler.to_dataset():
            p = self.p.prob(batch, batch_size=self.batch_size)
            q = self.q.prob(batch, batch_size=self.batch_size)
            p = np.clip(p, epsilon, math.inf)
            q = np.clip(q, epsilon, math.inf)
            m = 1 / 2 * (p + q)
            js = 1 / 2 * kl(p, m) + 1 / 2 * kl(q, m)
            js_sum += js
        return js_sum
