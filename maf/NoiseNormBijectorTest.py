from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras.keras_parameterized import TestCase
from tensorflow import Tensor

from common.util import set_seed
from maf.NoiseNormBijector import NoiseNormBijectorBuilder


class NoiseNormTest(TestCase):
    def init(self, normalise: bool, stddev: float):
        set_seed(42)
        self.builder = NoiseNormBijectorBuilder(normalise=normalise, noise_stddev=stddev)
        self.a: Tensor = tf.constant([[0, 1, 3], [0, 3, 2], [0, 5, 4], [0, 4, 2]], dtype=tf.float32)
        self.b: Tensor = tf.constant([[1, 2, 4], [0, 3, 2], [0, 5, 4], [0, 4, 2]], dtype=tf.float32)

    def test_det(self):
        b = NoiseNormBijectorBuilder(normalise=True)
        b.adapt(np.array([[0, 0], [4, 16]], dtype=np.float32))
        b.norm_shift = tf.zeros((1, 2), dtype=tf.float32)
        # b.norm_log_scale = tf.constant(tf.math.log([3.0, 2.0]), dtype=tf.float32)
        n = b.create()
        a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        print('original')
        print(a)
        forwarded = n.forward(a) * tf.exp(n.forward_log_det_jacobian(a))
        print('forwarded')
        print(forwarded)
        inverted = n.inverse(forwarded) * tf.exp(n.inverse_log_det_jacobian(forwarded))
        print('inversed again')
        print(inverted)
        self.assertAllEqual(a, inverted)

    def test_det2(self):
        b = NoiseNormBijectorBuilder(normalise=True)
        b.adapt(np.array([[0, 0], [4, 16]], dtype=np.float32))
        b.norm_shift = tf.zeros((1, 2), dtype=tf.float32)
        # b.norm_log_scale = tf.constant(tf.math.log([3.0, 2.0]), dtype=tf.float32)
        n = b.create()
        a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        print('original')
        print(a)
        inverted = n.inverse(a) * tf.exp(n.inverse_log_det_jacobian(a))
        print('inverted')
        print(inverted)
        forwarded = n.forward(inverted) * tf.exp(n.forward_log_det_jacobian(inverted))
        print('forwarded again')
        print(forwarded)
        self.assertAllEqual(a, forwarded)

    def test_norm_no_noise(self):
        self.init(normalise=True, stddev=0.0)
        self.builder.adapt(self.a)
        nn = self.builder.create()
        inversed = nn.inverse(self.b)
        print("111asd")

    def test_norm_with_noise(self):
        self.init(normalise=True, stddev=1.0)
        self.builder.adapt(self.a)
        nn = self.builder.create()
        inversed = nn.inverse(self.b)
        print("111asd")