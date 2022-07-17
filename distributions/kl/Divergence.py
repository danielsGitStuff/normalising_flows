from __future__ import annotations

import tensorflow as tf
from keras import backend as K

from common.jsonloader import Ser
from distributions.distribution import Distribution
from maf.DS import DS


class Divergence(Ser):
    def __init__(self, p: Distribution, q: Distribution, half_width: float, step_size: float, batch_size: int = 100000):
        super().__init__()
        self.p: Distribution = p
        self.q: Distribution = q
        self.dims: int = p.input_dim
        self.name: str = 'name not set'

        def check(d: Distribution):
            if d.conditional:
                raise RuntimeError('conditional KL not supported')
            if d.input_dim != self.dims:
                raise RuntimeError(f"expected input_dim is {self.dims} but a distribution has {d.input_dim}")

        check(p)
        check(q)
        self.half_width: float = half_width
        self.step_size: float = step_size
        self.batch_size: int = batch_size
        self.log_epsilon = tf.cast(tf.math.log(K.epsilon()), dtype=tf.float32)
        self.epsilon = tf.cast(K.epsilon(), dtype=tf.float32)

    def calculate_from_samples_vs_q(self, ds_p_samples: DS, log_p_samples: DS) -> float:
        raise NotImplementedError()

    def calculate_by_sampling_space(self) -> float:
        raise NotImplementedError()

    def calculate_by_sampling_p(self, no_of_samples: int) -> float:
        raise NotImplementedError()
