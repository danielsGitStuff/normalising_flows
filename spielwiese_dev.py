import numpy as np
from typing import List, Callable

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.monte_carlo as monte_carlo
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.base import TTensor
from distributions.kl.KL import KullbackLeiblerDivergence
from maf.DS import DS

if __name__ == '__main__':
    p = GaussianMultivariate(input_dim=1, mus=[1], cov=[1])
    q = GaussianMultivariate(input_dim=1, mus=[0], cov=[2 ** 2])
    p.create_base_distribution()
    q.create_base_distribution()
    true_kl = p.tfd_distribution.kl_divergence(q.tfd_distribution)
    print(f"true kl is {true_kl}")

    p_samples = p.sample(1000)
    log_ps_p_samples = p.log_prob(p_samples)
    log_ps_q_samples = q.log_prob(p_samples)
    ps_p_samples = np.exp(log_ps_p_samples)

    kl_mine =  (log_ps_p_samples - log_ps_q_samples)

    print(f"my {tf.reduce_mean(kl_mine)}")


    # true_log_ratio = p.log_prob(p_samples) - q.log_prob(p_samples)
    def true_log_ratio(xs: TTensor):
        return p.log_prob(xs) - q.log_prob(xs)


    def custom(xs: TTensor):
        log_r = q.log_prob(xs) - p.log_prob(xs)
        r = tf.exp(log_r)
        kl = (r - 1) - log_r
        return kl


    def custom2(xs: TTensor):
        return 1 / 2 * (q.log_prob(xs) - p.log_prob(xs)) ** 2


    def test(func: Callable, name: str):
        kl_divs: List[float] = []
        for i in range(50):
            p_samples = p.sample(5000)
            log_p_samples = p.log_prob(p_samples)
            kl = monte_carlo.expectation(f=func, samples=p_samples, log_prob=log_p_samples)
            kl_divs.append(kl)
        print(f"'{name}': avg kl {np.mean((kl_divs))}, std {np.std(kl_divs)}")


    kl1 = monte_carlo.expectation(f=true_log_ratio, samples=p_samples)
    print(f"kl1 {kl1}")

    own = KullbackLeiblerDivergence(p=p, q=q, half_width=1.0, step_size=1.0)
    kl_own = own.calculate_from_samples_vs_q(DS.from_tensor_slices(p_samples), log_p_samples=DS.from_tensor_slices(p.log_prob(p_samples)))
    print(f"kl_own {kl_own}")

    kl2 = monte_carlo.expectation(f=custom, samples=p_samples, log_prob=p.log_prob(p_samples))
    print(f"kl2 {kl2}")

    test(true_log_ratio, 'kl1')
    test(custom, 'kl2')
    test(custom2, 'kl2.5')
