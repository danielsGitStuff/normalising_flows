from typing import List

import numpy as np
from keras.keras_parameterized import TestCase

from distributions.ConditionalCategorical import ConditionalCategorical
from distributions.GaussianMultivariate import GaussianMultivariate


class T(TestCase):
    def test_categorical(self):
        c = ConditionalCategorical(input_dim=1, conditional_dims=1)
        c.add_d(d=GaussianMultivariate(input_dim=1, mus=[1], cov=[1]), weight=2)
        c.add_d(d=GaussianMultivariate(input_dim=1, mus=[-1], cov=[1]), weight=1)
        xs = np.array([[1.0], [-1.0], [0.0]], dtype=np.float32)
        cond = np.array([[0], [1], [0]], dtype=np.int)
        c.likelihoods(xs=xs, cond=cond)
        print('debug')

    def test_cat_and_cond(self):
        def produce_f(d: GaussianMultivariate, cond: List[float]) -> GaussianMultivariate:
            """just multiply Mu with the conditional value"""
            g = GaussianMultivariate(input_dim=d.input_dim, mus=[cond[0] * m for m in d.mus], cov=d.cov_matrix)
            return g

        g1 = GaussianMultivariate(input_dim=1, mus=[1], cov=[1])
        g2 = GaussianMultivariate(input_dim=1, mus=[-1], cov=[1])
        g1.make_conditional(conditional_dims=1, producer_function=produce_f)
        g2.make_conditional(conditional_dims=1, producer_function=produce_f)
        c = ConditionalCategorical(input_dim=1, conditional_dims=2, categorical_dims=1)
        c.add_d(d=g1, weight=2)
        c.add_d(d=g2, weight=1)
        xs = np.array([[1.0], [-1.0], [0.0]], dtype=np.float32)
        cond = np.array([[0, 2.0], [1, 2.0], [0, 2.0]], dtype=np.int)
        ls = c.likelihoods(xs=xs, cond=cond)
        expected: np.ndarray = np.array([[0.24197073],
                                         [0.24197073],
                                         [0.05399096]], dtype=np.float32)
        self.assertAllClose(expected, ls)
        print(ls)

    def test_cond(self):
        def produce_f(d: GaussianMultivariate, cond: List[float]) -> GaussianMultivariate:
            """just multiply Mu with the conditional value"""
            g = GaussianMultivariate(input_dim=d.input_dim, mus=[cond[0] * m for m in d.mus], cov=d.cov_matrix)
            return g

        g1 = GaussianMultivariate(input_dim=1, mus=[1], cov=[1])
        g1.make_conditional(conditional_dims=1, producer_function=produce_f)
        xs = np.array([[1.0], [-1.0], [0.0]], dtype=np.float32)
        cond = np.array([[2.0], [2.0], [2.0]], dtype=np.int)
        ls = g1.likelihoods(xs=xs, cond=cond)
        expected: np.ndarray = np.array([[0.24197073],
                                         [0.00443185],
                                         [0.05399096]], dtype=np.float32)
        self.assertAllClose(expected, ls)
        print(ls)
