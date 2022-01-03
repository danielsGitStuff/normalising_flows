from keras.keras_parameterized import TestCase

from common.util import Runtime
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.kl.KL import KullbackLeiblerDivergence


class KLTest(TestCase):
    def test_same_gaussians(self):
        print('same')
        dims: int = 2
        p = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        q = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        kld = KullbackLeiblerDivergence(p=p, q=q, half_width=6, step_size=.1)
        r = Runtime('kl').start()
        kl = kld.calculate()
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
        kl = kld.calculate()
        r.stop().print()
        print(f"kl is {kl}")
        self.assertNotEqual(kl, 0.0)
        self.assertGreater(kl, 0.0)
