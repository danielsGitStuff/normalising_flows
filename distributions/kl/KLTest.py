import numpy as np
from keras.keras_parameterized import TestCase

from common.util import Runtime
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.Uniform import Uniform
from distributions.UniformMultivariate import UniformMultivariate
from distributions.kl.KL import KullbackLeiblerDivergence


class KLTest(TestCase):
    def test_same_gaussians(self):
        print('same')
        dims: int = 2
        p = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        q = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        kld = KullbackLeiblerDivergence(p=p, q=q, half_width=6, step_size=.1)
        r = Runtime('kl').start()
        kl = kld.calculate_sample_space()
        r.stop().print()
        print(f"kl is {kl}")
        self.assertEqual(kl, 0.0)

    def test_different_gaussians(self):
        print('different')
        dims: int = 2
        p = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        q = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.79] * dims)
        kld = KullbackLeiblerDivergence(p=p, q=q, half_width=6, step_size=.1)
        r = Runtime('kl').start()
        kl = kld.calculate_sample_space()
        r.stop().print()
        print(f"kl is {kl}")
        self.assertNotEqual(kl, 0.0)
        self.assertGreater(kl, 0.0)

    def test_different_gaussians_tfp_1(self):
        print('different')
        dims: int = 1
        p = GaussianMultivariate(input_dim=dims, mus=[0] * dims, cov=[1.0] * dims)
        q = GaussianMultivariate(input_dim=dims, mus=[.1] * dims, cov=[1.0] * dims)

        p.create_base_distribution()
        q.create_base_distribution()

        kld_tf = float(p.tfd_distribution.kl_divergence(q.tfd_distribution))
        kld = KullbackLeiblerDivergence(p=p, q=q, half_width=6, step_size=.1)
        r = Runtime('kl').start()
        kl = kld.calculate_sample_distribution(1000000)
        # kl = kld.calculate_sample_space()
        r.stop().print()
        print(f"kl is {kl}. tf_kl is {kld_tf}")
        self.assertNotEqual(kl, 0.0)
        self.assertAlmostEqual(kl, kld_tf, places=3)

    def test_different_gaussians_tfp_2(self):
        print('different')
        dims: int = 1
        p = GaussianMultivariate(input_dim=dims, mus=[0] * dims, cov=[1.0] * dims)
        q = GaussianMultivariate(input_dim=dims, mus=[1] * dims, cov=[1.0] * dims)

        p.create_base_distribution()
        q.create_base_distribution()

        kld_tf = float(p.tfd_distribution.kl_divergence(q.tfd_distribution))
        kld = KullbackLeiblerDivergence(p=p, q=q, half_width=6, step_size=.1)
        r = Runtime('kl').start()
        kl = kld.calculate_sample_distribution(1000)
        # kl = kld.calculate_sample_space()
        r.stop().print()
        print(f"kl is {kl}. tf_kl is {kld_tf}")
        self.assertNotEqual(kl, 0.0)
        self.assertLessEqual(np.absolute(kl - kld_tf), 0.006)
        # self.assertAlmostEqual(kl, kld_tf, places=2)
