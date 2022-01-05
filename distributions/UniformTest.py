from __future__ import annotations

import numpy as np
from tensorflow_datasets.testing import TestCase

from distributions.Uniform import Uniform


class UniformTest(TestCase):
    def setUp(self):
        self.u = Uniform(low=1.0, high=2.5)
        self.a: np.ndarray = np.array([[0.1], [1.0], [1.1], [2.0]], dtype=np.float32)

    def test_ll(self):
        ll = self.u.likelihoods(self.a)
        self.assertAllEqual(ll, np.array([[0.0], [2 / 3], [2 / 3], [2 / 3]], dtype=np.float32))