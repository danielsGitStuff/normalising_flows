from __future__ import annotations

import tensorflow as tf
from typing import Tuple

from distributions.Distribution import Distribution
from distributions.base import TTensor, BaseMethods
from distributions.kl.KL import KullbackLeiblerDivergence
from distributions.kl.KLSampler import KLSampler
from maf.DS import DS


class JensenShannonDivergence(KullbackLeiblerDivergence):
    class Methods:
        @staticmethod
        def js_tensor_parts(log_p: TTensor, log_q: TTensor) -> Tuple[float, float]:
            """
            Calculates m, KL(p, m) and KL(q,m)
            Make sure that neither log_p nor log_q contain any -Inf values. This will cause NaNs and Infs down the road!
            Use BaseMethods.filter_log_space_neg_inf(log_p, log_q) to filter it before calling this.
            @return kl_p_m, kl_q_m"""

            if len(log_p) == 0:
                return 0.0, 0.0

            p = tf.exp(log_p)
            q = tf.exp(log_q)
            log_m = tf.math.log(1 / 2 * (p + q))
            kl_p_m, _ = KullbackLeiblerDivergence.Methods.kl_tensors(log_p, log_m)
            kl_q_m, _ = KullbackLeiblerDivergence.Methods.kl_tensors(log_q, log_m)
            return kl_p_m, kl_q_m

    def __init__(self, p: Distribution, q: Distribution, half_width: float, step_size: float, batch_size: int = 100000):
        super().__init__(p, q, half_width, step_size, batch_size)
        self.name='js'

    def calculate_by_sampling_space(self) -> float:
        raise NotImplementedError("not adapted to the new stuff yet")
        sampler = KLSampler(dims=self.dims, half_width=self.half_width, step_size=self.step_size, batch_size=self.batch_size)
        print(f"will calculate JS divergence in {len(sampler.batch_sizes)} batches")
        js_sum: float = 0.0

        for batch in sampler.to_dataset():
            js_sum += self.js_batch(batch)
        return js_sum

    def calculate_by_sampling_p(self, no_of_samples: int) -> float:
        left: int = no_of_samples
        kl_sum_p_m: float = 0.0
        kl_sum_q_m: float = 0.0
        samples_sum: int = 0
        while left > 0:
            take: int = min(left, self.batch_size)
            batch = self.p.sample(take)
            log_p = self.p.log_prob(batch, batch_size=self.batch_size)
            log_q = self.q.log_prob(batch, batch_size=self.batch_size)
            log_p, log_q = BaseMethods.filter_log_space_neg_inf(log_p, log_q)
            kl_p_m, kl_q_m = JensenShannonDivergence.Methods.js_tensor_parts(log_p=log_p, log_q=log_q)
            kl_sum_p_m += kl_p_m
            kl_sum_q_m += kl_q_m
            samples_sum += len(log_p)
            left -= take
        kl_sum_p_m = kl_sum_p_m / samples_sum
        kl_sum_q_m = kl_sum_q_m / samples_sum
        js_sum = (kl_sum_p_m + kl_sum_q_m) / 2
        return js_sum

    def calculate_from_samples_vs_p(self, ds_q_samples: DS, log_q_samples: DS) -> float:
        kl_sum_p_m: float = 0.0
        kl_sum_q_m: float = 0.0
        samples_sum: int = 0
        for samples, log_q in zip(ds_q_samples, log_q_samples):
            log_p = self.p.log_prob(samples, batch_size=self.batch_size)
            log_p, log_q = BaseMethods.filter_log_space_neg_inf(log_p, log_q)
            kl_p_m, kl_q_m = JensenShannonDivergence.Methods.js_tensor_parts(log_p=log_p, log_q=log_q)
            kl_sum_p_m += kl_p_m
            kl_sum_q_m += kl_q_m
            samples_sum += len(log_p)
        kl_sum_p_m = kl_sum_p_m / samples_sum
        kl_sum_q_m = kl_sum_q_m / samples_sum
        js_sum = (kl_sum_p_m + kl_sum_q_m) / 2
        return js_sum
