from typing import List

import numpy as np
from keras.keras_parameterized import TestCase

from common.globals import Global
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.UniformMultivariate import UniformMultivariate
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate


class WeightedMultimodalMultivariateTest(TestCase):
    def setUp(self):
        d = WeightedMultimodalMultivariate(input_dim=2)
        d.add_d(d=UniformMultivariate(input_dim=2, lows=[-1, -1], highs=[0, 0]), weight=1.0)
        d.add_d(d=UniformMultivariate(input_dim=2, lows=[0, 0], highs=[1, 1]), weight=2.0)
        self.d: WeightedMultimodalMultivariate = d

    def test_pro_of_samples(self):
        samples = self.d.sample(10)
        ps = self.d.prob(samples)
        log_ps = self.d.log_prob(samples)
        self.assertAllEqual(ps, np.exp(log_ps))
        print('ende')

    def test_pro_of_samples_1d(self):
        d = WeightedMultimodalMultivariate(input_dim=1)
        d.add_d(d=UniformMultivariate(input_dim=1, lows=[-1, ], highs=[0]), weight=1.0)
        d.add_d(d=UniformMultivariate(input_dim=1, lows=[0], highs=[1]), weight=2.0)
        self.d: WeightedMultimodalMultivariate = d

        p1 = d.prob([.1])[0]
        p2 = d.prob([-.1])[0]
        self.assertAlmostEqual(2 / 3, p1)
        self.assertAlmostEqual(1 / 3, p2)

    def test_gaussian_1d(self):
        g1 = GaussianMultivariate(input_dim=1, mus=[-2], cov=[.1])
        g2 = GaussianMultivariate(input_dim=1, mus=[0], cov=[1])
        g3 = GaussianMultivariate(input_dim=1, mus=[2], cov=[.2])
        data_distr = WeightedMultimodalMultivariate(input_dim=1)
        data_distr.add_d(g1, weight=1)
        data_distr.add_d(g2, weight=1)
        data_distr.add_d(g3, weight=2)

        m1 = 1 / 4
        m2 = 1 / 4
        m3 = 2 / 4

        x1 = [0.2]
        p1 = g1.prob(x1)
        p2 = g2.prob(x1)
        p3 = g3.prob(x1)

        p123 = p1 * m1 + p2 * m2 + p3 * m3
        p123 = p123 / 4

        p = data_distr.prob(x1)

        print('asd')

