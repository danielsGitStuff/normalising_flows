from __future__ import annotations

import math
from typing import Callable, List, Union, Optional

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from distributions.base import cast_to_tensor, cast_to_ndarray, TTensorOpt
from distributions.Distribution import Distribution
from distributions.GaussianMultiModal import GaussianMultimodal
from distributions.GaussianMultivariate import GaussianMultivariate


class NGaussianTest(tf.test.TestCase):
    def setUp(self):
        tf.compat.v1.enable_eager_execution(
            config=None, device_policy=None, execution_mode=None
        )

        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

        print(f"EAGER: {tf.executing_eagerly()}")
        from distributions.MultimodalDistribution import MultimodalDistribution
        self.n_g = MultimodalDistribution(input_dim=3, distributions=[GaussianMultivariate(3, mus=[2.0, 2.0, 1.0], cov=[1.0, 1.0, 1.0])])
        self.input_array = np.array([[2.0, 2.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        self.gaussian2d = Multiple2dGaussians(gaussians=[Gaussian2D(mu_x=-3.0, mu_y=-3.0, sigma=1.0), Gaussian2D(mu_x=3.0, mu_y=3.0, sigma=1.0)])
        self.gaussianND = MultimodalDistribution(2, distributions=[GaussianMultivariate(input_dim=2, mus=[-3.0, -3.0], cov=[1.0, 1.0]),
                                                                   GaussianMultivariate(input_dim=2, mus=[3.0, 3.0], cov=[1.0, 1.0])])

    def test_n(self):
        likelihoods = self.n_g.prob(self.input_array)
        self.assertAllClose([0.0634, 0.0141], likelihoods[:, 0], atol=0.01)

    def test_with_2d(self):
        ins = np.array([[-3.0, -3.0], [3.0, 3.0], [-2.7, 4.0]])
        ls2d = self.gaussian2d.likelihoods(ins)
        lsnd = self.gaussianND.likelihoods(ins)
        self.assertAllClose(ls2d, lsnd)
        self.assertEqual(ls2d[0], ls2d[1])

    def test_logprob_vs_prob(self):
        ins = np.array([[-3.0, -3.0], [3.0, 3.0], [-2.7, 4.0]])
        log_lh = self.gaussianND.log_likelihoods(ins)
        lh = self.gaussianND.likelihoods(ins)


class Multiple2dGaussians(Distribution):
    def __init__(self, gaussians: Optional[List[Gaussian2D]] = None):
        super().__init__(input_dim=2)
        if gaussians is None:
            gaussians = [Gaussian2D(mu_x=-3.0, mu_y=-3.0, sigma=1.0), Gaussian2D(mu_x=3.0, mu_y=3.0, sigma=1.0)]
        # dist specific
        self.gaussians: List[Gaussian2D] = gaussians

    def sample(self, size: int = 1) -> np.ndarray:
        vss = []
        gaussian_idx = np.random.choice(self.gaussians, size)
        for gaussian in gaussian_idx:
            vs = gaussian.sample(size=1)
            # vs = np.concatenate([vs, [1.0]])
            vss.append(vs)
        vss = np.array(vss)
        return vss

    def likelihoods(self, xs: Union[Tensor, np.ndarray]) -> np.ndarray:
        if not isinstance(xs, Tensor):
            xs = tf.constant(xs, dtype=tf.float32)
        x = xs[:, 0]
        y = xs[:, 1]

        ls: List[np.ndarray] = [g.likelihoods(xs) for g in self.gaussians]
        ls: np.ndarray = np.stack(ls, axis=1)
        # likelihood = tf.reduce_sum(ls, axis=1) / len(self.gaussians)
        likelihood = np.sum(ls, axis=1) / len(self.gaussians)
        return self.cast_2_likelihood(input_tensor=xs, result=likelihood)

    def prob(self, xs: Union[np.ndarray, Tensor]) -> np.ndarray:
        """convenience function to print a likelihood map"""
        if not isinstance(xs, Tensor):
            xs = tf.constant(xs, dtype=tf.float32)
        likelihoods = self.likelihoods(xs)
        return self.cast_2_likelihood(input_tensor=xs, result=likelihoods)


class Gaussian(Distribution):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__(input_dim=1)
        self.mu: float = mu
        self.sigma: float = sigma
        self.tf_mu: Tensor = tf.constant(mu)
        self.tf_sigma: Tensor = tf.constant(sigma)
        self.PI: Tensor = tf.constant(math.pi)
        self.pdf_o \
            : Callable[[float], float] = lambda x: 1 / (self.sigma * math.sqrt(2 * math.pi)) * math.pow(math.e, -1 / 2 * ((x - self.mu) / self.sigma) ** 2)
        self.pdf: Callable[[Tensor], Tensor] = lambda x: 1 / (self.tf_sigma * tf.sqrt(2 * self.PI)) * tf.exp(-1 / 2 * tf.pow((x - self.tf_mu) / self.tf_sigma, 2))

    def sample(self, size: int = 1) -> np.ndarray:
        xs = np.random.normal(loc=self.mu, scale=self.sigma, size=size)
        return xs

    def likelihoods(self, x: Tensor) -> np.ndarray:
        tensor = self.pdf(x)
        return cast_to_ndarray(tensor)


class Gaussian2D(Distribution):
    def __init__(self, mu_x: float, mu_y: float, sigma: float = 1.0):
        super().__init__(input_dim=2)
        self.gauss_x: Gaussian = Gaussian(mu=mu_x, sigma=sigma)
        self.gauss_y: Gaussian = Gaussian(mu=mu_y, sigma=sigma)

    def sample(self, size: int = 1) -> np.ndarray:
        xs = self.gauss_x.sample(size)
        ys = self.gauss_y.sample(size)
        result = np.concatenate([xs, ys])
        return result.reshape((size, 2))

    def likelihoods(self, xs: Tensor) -> np.ndarray:
        # geometric interpretation. only works when sigma is the same for both gaussians
        # delta_x = x - self.gauss_x.mu
        # delta_y = y - self.gauss_y.mu
        # d = np.sqrt(delta_x ** 2 + delta_y ** 2)
        # normal = Gaussian()
        # likelihood = normal.likelihood(d) ** 2
        assert xs.shape[1] == 2
        x = xs[:, 0]
        y = xs[:, 1]

        lx = self.gauss_x.likelihoods(x)
        ly = self.gauss_y.likelihoods(y)
        likelihood = lx * ly
        return self.cast_2_likelihood(input_tensor=xs, result=likelihood)
