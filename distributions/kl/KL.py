from __future__ import annotations

import math
import tensorflow as tf
from keras import backend as K
from typing import Tuple

from distributions.Distribution import Distribution
from distributions.base import TTensor, BaseMethods
from distributions.kl.KLSampler import KLSampler


class KullbackLeiblerDivergence:

    def __init__(self, p: Distribution, q: Distribution, half_width: float, step_size: float, batch_size: int = 100000):

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
        self.log_epsilon = tf.cast(tf.math.log(K.epsilon()), dtype=tf.float32)
        self.epsilon = tf.cast(K.epsilon(), dtype=tf.float32)

    def kl_tensors(self, log_p: TTensor, log_q: TTensor) -> Tuple[float, int]:
        """calculate the Kullback-Leibler-Divergence for one batch
        @return kl divergence and the amount of valid samples used"""

        log_p, log_q = BaseMethods.filter_log_space_neg_inf(log_p, log_q)

        # kl = tf.reduce_sum(p * (log_p - log_q))  # vanilla KL, does not work

        # kl = tf.reduce_sum(log_p - log_q)  # naive version k1, high variance, unbiased, src=http://joschu.net/blog/kl-approx.html
        # kl = tf.reduce_sum(1 / 2 * (log_p - log_q) ** 2) # k2, low variance, biased

        log_r = log_q - log_p  # k3, low variance, unbiased
        r = tf.exp(log_r)
        kl = tf.reduce_sum((r - 1) - log_r)
        # kl = tf.reduce_sum(r * log_r - (r - 1))  # k3 reverse, low variance, unbiased

        # r = p / q  # approx seems to work, src=https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd
        # log_r = log_p - log_q  # also equals k3
        # kl = tf.reduce_sum(r * log_r - (r - 1))

        return float(kl), len(log_p)

    def kl_batch(self, batch: TTensor) -> Tuple[float, int]:
        log_p = self.p.log_prob(batch, batch_size=self.batch_size)
        log_q = self.q.log_prob(batch, batch_size=self.batch_size)
        return self.kl_tensors(log_q=log_q, log_p=log_p)
        log_p = K.clip(log_p, tf.float32.min, 0.0)
        log_q = K.clip(log_q, tf.float32.min, 0.0)

        # kl = tf.reduce_sum(p * (log_p - log_q))  # vanilla KL, does not work

        # ALTERNATIVE URL: https://web.archive.org/web/20220128194513/https://joschu.net/blog/kl-approx.html
        # kl = tf.reduce_sum(log_p - log_q)  # naive version k1, high variance, unbiased, src=http://joschu.net/blog/kl-approx.html
        # kl = tf.reduce_sum(1 / 2 * (log_p - log_q) ** 2) # k2, low variance, biased

        log_r = log_q - log_p  # k3, low variance, unbiased
        r = tf.exp(log_r)
        kl = tf.reduce_sum((r - 1) - log_r)
        # kl = tf.reduce_sum(r * log_r - (r - 1))  # k3 reverse, low variance, unbiased

        # r = p / q  # approx seems to work, src=https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd
        # log_r = log_p - log_q  # also equals k3
        # kl = tf.reduce_sum(r * log_r - (r - 1))

        return float(kl)

    def calculate_sample_space(self) -> float:
        sampler = KLSampler(dims=self.dims, half_width=self.half_width, step_size=self.step_size, batch_size=self.batch_size)
        print(f"will calculate KL divergence in {len(sampler.batch_sizes)} batches")
        kl_sum: float = 0.0
        samples_sum: int = 0
        for batch in sampler.to_dataset():
            kl, no_of_samples = self.kl_batch(batch)
            samples_sum += no_of_samples
            kl_sum += kl
        return kl_sum / samples_sum

    def calculate_sample_distribution(self, no_of_samples: int) -> float:
        left: int = no_of_samples
        kl_sum: float = 0.0
        samples_sum: int = 0
        while left > 0:
            take = min(left, self.batch_size)
            batch = self.p.sample(take)
            kl, no_of_samples = self.kl_batch(batch)
            kl_sum += kl
            samples_sum += no_of_samples
            left -= take

        kl = kl_sum / samples_sum
        return kl
