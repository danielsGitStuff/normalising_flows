from __future__ import annotations

from typing import Optional

import numpy as np
from keras.losses import KLDivergence

from distributions.Distribution import Distribution
from distributions.kl.KLSampler import KLSampler
import tensorflow as tf
from keras import backend as K


class KullbackLeiblerDivergence:
    def __init__(self, p: Distribution, q: Distribution, half_width: float, step_size: float, batch_size: int = 100000 ):

        self.p: Distribution = p
        self.q: Distribution = q
        self.dims: int = p.input_dim

        def check(d: Distribution):
            if d.conditional:
                raise RuntimeError('conditional KL not supported')
            if d.input_dim != self.dims:
                raise RuntimeError(f"expected input_dim is {self.dims} but a distribution has {d.input_dim}")

        check(p)
        check(q)
        self.half_width: float = half_width
        self.step_size: float = step_size
        self.batch_size: int = batch_size

    def calculate(self) -> float:
        sampler = KLSampler(dims=self.dims, half_width=self.half_width, step_size=self.step_size, batch_size=self.batch_size)
        print(f"will calculate KL divergence in {len(sampler.batch_sizes)} batches")
        kl_sum: float = 0.0
        # kl_debug: float = 0.0
        log_epsilon = tf.cast(tf.math.log(K.epsilon()), dtype=tf.float32)
        for batch in sampler.to_dataset():
            log_p = self.p.log_prob(batch, batch_size=self.batch_size)
            log_q = self.q.log_prob(batch, batch_size=self.batch_size)
            log_p = K.clip(log_p, log_epsilon, 0.0)
            log_q = K.clip(log_q, log_epsilon, 0.0)
            p = np.exp(log_p)
            # q = np.exp(log_q)
            kl = tf.reduce_sum(p * (log_p - log_q))
            # k = KLDivergence()
            # debug = k(p, q)
            # kl_debug += debug
            kl_sum += kl
        return float(kl_sum)
