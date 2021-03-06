from typing import Optional, Tuple

import tensorflow as tf
# from tensorflow_probability.python import bijectors as tfb
import tensorflow_probability.python.bijectors as tfb


class CustomMade(tf.keras.layers.Layer):
    """
    Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
    The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
    and one log_scale vector.

    :param params: Python integer specifying the number of parameters to output per input.
    :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
    :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
    :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
    :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
    """

    def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made", kernel_initializer=None, conditional: bool = False,
                 conditional_event_shape: Optional[Tuple[int]] = None):
        super(CustomMade, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.conditional: bool = conditional
        self.conditional_event_shape = conditional_event_shape

        self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
                                                 activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                                                 bias_regularizer=bias_regularizer, conditional=conditional, conditional_event_shape=conditional_event_shape)

    def call(self, x, **kwargs):
        shift, log_scale = tf.unstack(self.network(x, **kwargs), num=2, axis=-1)
        # return shift, log_scale
        return shift, tf.math.tanh(log_scale)  # original
        return shift, tf.math.sigmoid(log_scale)
