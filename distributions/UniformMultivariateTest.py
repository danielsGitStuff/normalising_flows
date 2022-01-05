import numpy as np
from keras.keras_parameterized import TestCase

from distributions.UniformMultivariate import UniformMultivariate


class UniformMultivariateTest(TestCase):
    def setUp(self):
        self.u = UniformMultivariate(input_dim=2, lows=[1.0, 2.0], highs=[2.5, 2.5])
        self.a: np.ndarray = np.array([[0.0, 2.0], [1.0, 2.0]])

    def test_a(self):
        ll = self.u.likelihoods(self.a)
        self.assertAllEqual(ll, np.array([[0.0], [4 / 3]], dtype=np.float32))

    def test_sample(self):
        samples = self.u.sample(3)
        print(samples)