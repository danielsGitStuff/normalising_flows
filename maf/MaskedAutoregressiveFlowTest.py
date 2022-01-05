from __future__ import annotations

import os
import shutil

import keras.layers
import numpy as np
from keras.keras_parameterized import TestCase
from numpy import ufunc
from tensorflow_probability.python.distributions import TransformedDistribution

from common.util import Runtime
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow


class MafTest(TestCase):
    def setUp(self):
        self.dim: int = 2
        self.maf: MaskedAutoregressiveFlow = MaskedAutoregressiveFlow(input_dim=self.dim, layers=4, hidden_shape=[5, 5], norm_layer=True)
        self.ones = np.ones(shape=(3, 2))
        self.zeros = np.zeros(shape=(3, 2))

    def check_tdfs(self, original: TransformedDistribution, cp: TransformedDistribution, op: ufunc = np.equal):
        """check whether two TDFs produce the same log likelihoods"""
        print(f"check ones and zeros with op '{op.__name__}'")
        for ar in [self.ones, self.zeros]:
            p: np.ndarray = original.log_prob(ar).numpy()
            p_copy: np.ndarray = cp.log_prob(ar).numpy()
            print(f"{p} vs {p_copy}")
            assert np.all(op(p, p_copy))

    def test_after_fit(self):
        xs = np.random.random_sample(self.dim * 10).reshape((10, self.dim)).astype(np.float32) + 2
        rt = Runtime(name="deepcopy").start()
        self.maf.adapt(xs)
        before = self.maf.log_prob(xs)
        tdf: TransformedDistribution = self.maf.transformed_distribution
        before_ones = tdf.log_prob(self.ones)
        maf_copy: MaskedAutoregressiveFlow = MaskedAutoregressiveFlow.Methods.deepcopy_maf(self.maf)
        tdf_copy = maf_copy.transformed_distribution
        rt.stop().print()
        print("before fit")

        dense_layers = [d for d in tdf.submodules if isinstance(d, keras.layers.core.Dense)]
        weights = [d.get_weights() for d in dense_layers]

        self.check_tdfs(self.maf.transformed_distribution, tdf_copy, op=np.equal)
        self.maf.fit(dataset=xs, batch_size=10, epochs=2, lr=0.1)

        dense_layers_after_fit = [d for d in tdf.submodules if isinstance(d, keras.layers.core.Dense)]
        weights_after_fit = [d.get_weights() for d in dense_layers_after_fit]

        dense_layers_copy = [d for d in tdf_copy.submodules if isinstance(d, keras.layers.core.Dense)]
        weights_copy = [d.get_weights() for d in dense_layers_copy]

        print("after fit")
        self.check_tdfs(self.maf.transformed_distribution, tdf_copy, op=np.not_equal)

    def test_after_save_and_load(self):
        test_dir = "test_dir"
        os.makedirs(test_dir, exist_ok=True)

        xs = np.random.random_sample(self.dim * 10).reshape((10, self.dim)).astype(np.float32)
        self.maf.fit(dataset=xs, batch_size=10, epochs=2)

        self.maf.save(test_dir, "bla")
        loaded: MaskedAutoregressiveFlow = LearnedTransformedDistribution.load(test_dir, "bla")
        self.check_tdfs(self.maf.transformed_distribution, loaded.transformed_distribution, op=np.equal)
        shutil.rmtree(test_dir)