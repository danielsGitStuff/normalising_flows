from keras.keras_parameterized import TestCase

from common.util import Runtime
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.kl.JS import JensenShannonDivergence
from distributions.kl.KL import KullbackLeiblerDivergence


class JSTest(TestCase):
    def test_same_gaussians(self):
        print('same')
        dims: int = 2
        p = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        q = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        kld = JensenShannonDivergence(p=p, q=q, half_width=6, step_size=.1)
        r = Runtime('kl').start()
        js = kld.calculate_by_sampling_space()
        r.stop().print()
        print(f"js is {js}")
        self.assertEqual(js, 0.0)

    def test_different_gaussians(self):
        print('different')
        dims: int = 2
        p = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.8] * dims)
        q = GaussianMultivariate(input_dim=dims, mus=[-.2] * dims, cov=[0.79] * dims)
        kld = JensenShannonDivergence(p=p, q=q, half_width=6, step_size=.1)
        r = Runtime('kl').start()
        js = kld.calculate_by_sampling_space()
        r.stop().print()
        print(f"js is {js}")
        self.assertNotEqual(js, 0.0)
        self.assertGreater(js, 0.0)
