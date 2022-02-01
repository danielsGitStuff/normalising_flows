from typing import List

import numpy as np
from keras.keras_parameterized import TestCase

from common.globals import Global
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
