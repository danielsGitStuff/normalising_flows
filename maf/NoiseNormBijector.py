from __future__ import annotations

import math
from typing import Optional

import numpy as np
# todo revert import hackery
from tensorflow.python.keras.layers import Normalization
# from keras.layers import Normalization
from tensorflow import Tensor
from tensorflow_probability.python.bijectors import Bijector
import tensorflow as tf
from common.jsonloader import Ser
from distributions.base import cast_to_ndarray
import keras.backend as K

from maf.DS import DS


class NoiseNormBijectorBuilder(Ser):
    class Methods:
        @staticmethod
        def add_noise(xs: Tensor, stddev: float):
            return xs + tf.random.normal(xs.shape, mean=0.0, stddev=stddev)

    def __init__(self, normalise: bool = False, noise_stddev: float = 0.0):
        super().__init__()
        self.normalise: bool = normalise
        self.noise_stddev: float = noise_stddev
        self.norm_shift: Optional[np.ndarray] = None
        self.norm_log_scale: Optional[np.ndarray] = None
        self.norm_scale: Optional[np.ndarray] = None
        # if self.normalise and self.noise_stddev <= 0.0:
        #     raise RuntimeError()

    def create(self) -> NoiseNormBijector:
        return NoiseNormBijector(normalise=self.normalise, norm_log_scale=self.norm_log_scale, norm_shift=self.norm_shift, noise_stddev=self.noise_stddev,
                                 norm_scale=self.norm_scale)

    def adapt(self, ds: DS):
        if self.normalise:
            n: Normalization = Normalization()
            batch_size = None
            if len(ds) > 10000:
                batch_size = math.floor(len(ds) / 10000)
                batch_size = min(batch_size, 10000)
            n.adapt(ds, batch_size=batch_size)
            self.norm_shift = cast_to_ndarray(n.mean)
            stddev = tf.sqrt(n.variance)
            # self.scale = tf.maximum(self.scale, K.epsilon())
            min_stddev = self.noise_stddev if self.noise_stddev > 0.0 else K.epsilon()
            min_stddev = max(min_stddev, K.epsilon())
            self.norm_scale = cast_to_ndarray(tf.maximum(stddev, min_stddev))
            self.norm_log_scale = cast_to_ndarray(tf.math.log(tf.maximum(stddev, min_stddev)))


class NoiseNormBijector(Bijector):
    def __init__(self, normalise: bool = False, norm_log_scale: Optional[Tensor] = None, norm_scale: Optional[Tensor] = None, norm_shift: Optional[Tensor] = None,
                 noise_stddev: float = 0.0):
        super(NoiseNormBijector, self).__init__(is_constant_jacobian=True, inverse_min_event_ndims=0, name='noise_norm')
        self.normalise: bool = normalise
        self.norm_log_scale: Optional[Tensor] = norm_log_scale
        self.norm_shift: Optional[Tensor] = norm_shift
        self.noise_stddev: float = noise_stddev
        self.norm_scale: Optional[Tensor] = norm_scale

    def _inverse(self, y, training: bool = False):
        if self.normalise:
            if self.noise_stddev > 0.0 and training:
                y = y + tf.random.normal(y.shape, mean=0.0, stddev=self.noise_stddev)
            # scale = tf.exp(self.norm_log_scale)
            scale = self.norm_scale
            return (y - self.norm_shift) / scale
        else:
            if self.noise_stddev > 0.0 and training:
                y = y + tf.random.normal(y.shape, mean=0.0, stddev=self.noise_stddev)
                return y
        return y

    def _forward(self, x, training: bool = False):
        if self.normalise:
            # scale = tf.exp(self.norm_log_scale)
            scale = self.norm_scale
            return x * scale + self.norm_shift
        return x

    def inverse_log_det_jacobian(self,
                                 y,
                                 event_ndims=None,
                                 name='inverse_log_det_jacobian',
                                 **kwargs):
        if self.normalise:
            r = -tf.reduce_sum(self.norm_log_scale)
            # r = tf.math.log(tf.abs(1 / tf.reduce_prod(self.norm_scale)))
        else:
            # r = tf.zeros(shape=(y.shape[0], 1))
            r = tf.constant(0.0, dtype=tf.float32)
        return r

    # def forward_log_det_jacobian(self,
    #                              x,
    #                              event_ndims=None,
    #                              name='forward_log_det_jacobian',
    #                              **kwargs):
    #     if self.normalise:
    #         r = tf.reduce_sum(self.norm_log_scale)
    #         # r = tf.math.log(tf.abs(tf.reduce_prod(self.norm_scale)))
    #     else:
    #         # r = tf.zeros(shape=(x.shape[0], 1))
    #         r = tf.constant(0.0, dtype=tf.float32)
    #     return r
    # THESE COMMENTED OUT METHODS RESIDE HERE AS A MEMORIAL:
    # THESE METHODS DO NOT CALCULATE THE ACTUAL LOG DETERMINANT, DESPITE THEIR NAME SUGGESTING EXACTLY THAT.
    # IN A MAF THEY MUST RETURN THE LOG OF THE DIAGONAL OF THE JACOBIAN MATRIX, NOT THE DETERMINANT!
    # Play around and find out yourself. tf.__version__ = '2.7.0', tf_probability.__version__ = '0.14.1'
    # def _inverse_log_det_jacobian(self, y, **kwargs):
    #     if self.normalise:
    #         # r = tf.reduce_sum(self.norm_log_scale)
    #         # r = tf.math.log(1/tf.reduce_prod(self.norm_scale))
    #         # r = tf.math.log(tf.abs(1 / tf.reduce_prod(self.norm_scale)))
    #         r = -self.norm_log_scale
    #     else:
    #         r = tf.zeros(shape=(y.shape[0], 1))
    #         r = tf.constant(0.0, dtype=tf.float32)
    #     return r
    #
    # def _forward_log_det_jacobian(self, x, **kwargs):
    #     if self.normalise:
    #         # r = -tf.reduce_sum(self.norm_log_scale)
    #         # r = tf.math.log(tf.reduce_prod(self.norm_scale))
    #         # r = tf.math.log(tf.abs(tf.reduce_prod(self.norm_scale)))
    #         r = self.norm_log_scale
    #     else:
    #         r = tf.zeros(shape=(x.shape[0], 1))
    #         r = tf.constant(0.0, dtype=tf.float32)
    #     return r


