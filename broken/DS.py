from __future__ import annotations

from typing import Optional, Tuple, Callable, Collection, Union

import tensorflow as tf
from tensorflow import Tensor

from common import jsonloader
from common.jsonloader import Ser
from distributions.base import enable_memory_growth, TTensor
from common.globals import Global

DS = tf.data.Dataset
DSOpt = Optional[tf.data.Dataset]


class DSMethods:
    @staticmethod
    def extract_xs_cond(ds: DS, conditional_dims: int) -> Tuple[DS, DSOpt]:
        if conditional_dims == 0:
            return ds, None
        cond = lambda t: t[:conditional_dims]

    @staticmethod
    def batch(d: DSOpt, batch_size: Optional[int]) -> DSOpt:
        if d is None:
            return None
        if batch_size is None:
            return d.batch(len(d))
        return d.batch(batch_size)


class DatasetProps(Ser):
    def __init__(self):
        super().__init__()
        self.length: int = None
        self.shape: Tuple[int, int] = None
        self.dimensions: int = None
        self.conditional_dimensions = None
        self.classes: Collection[Union[int, str]] = None
        self.no_of_signals: int = None
        self.no_of_noise: int = None
        self.no_of_columns: int = None


class DataLoader(Ser):
    class Methods:
        @staticmethod
        def logit(xs, ys, beta=10e-6):
            """
            Conversion to logit space according to equation (24) in [Papamakarios et al. (2017)].
            Includes scaling the input image to [0, 1] and conversion to logit space.
            :param xs: Input tensor, e.g. image. Type: tf.float32.
            :param beta: Small value. Default: 10e-6.
            :return: Input tensor in logit space.
            """
            inter = beta + (1 - 2 * beta) * (xs / 256)
            return tf.math.log(inter / (1 - inter)), ys  # logit function

        @staticmethod
        def logit_xs(xs, beta=10e-6):
            """
            Conversion to logit space according to equation (24) in [Papamakarios et al. (2017)].
            Includes scaling the input image to [0, 1] and conversion to logit space.
            :param xs: Input tensor, e.g. image. Type: tf.float32.
            :param beta: Small value. Default: 10e-6.
            :return: Input tensor in logit space.
            """
            inter = beta + (1 - 2 * beta) * (xs / 256)
            return tf.math.log(inter / (1 - inter))  # logit function

        @staticmethod
        def inverse_logit(xs, beta=10e-6):
            """
            Reverts the preprocessing steps and conversion to logit space and outputs an image in
            range [0, 256]. Inverse of equation (24) in [Papamakarios et al. (2017)].
            :param x: Input tensor in logit space. Type: tf.float32.
            :param beta: Small value. Default: 10e-6.
            :return: Input tensor in logit space.
            """

            x = tf.math.sigmoid(xs)
            return (x - beta) * 256 / (1 - 2 * beta)

        @staticmethod
        def normalize_img(image, label, div: float = 255):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / div, label

        @staticmethod
        def normalize_img_minus1_1() -> Callable[[Tensor, Tensor], Tensor]:
            def normalize_img(image, label):
                """Normalizes images: `uint8` -> `float32`."""
                half = tf.constant(255 / 2, dtype=tf.float32)
                return (tf.cast(image, tf.float32) - half) / 255 * 2, label

            return normalize_img

        @staticmethod
        def run_in_process(js: str):
            print("running in process")
            enable_memory_growth()
            # print(js)
            # print(tf.config.list_physical_devices())
            dl: DataLoader = jsonloader.from_json(js)
            dl.create_data_sets()
            print("done with process")
            # sys.exit(11)

    def __init__(self, conditional: bool = False):
        super().__init__()
        self.conditional: bool = conditional
        self.no_of_signals: Optional[int] = None
        self.no_of_noise: Optional[int] = None

    def init(self):
        js = jsonloader.to_json(self, pretty_print=True)
        # DataLoader.Methods.run_in_process(js)
        Global.POOL().run_blocking(DataLoader.Methods.run_in_process, args=(js,))

    def create_data_sets(self, load_limit: Optional[int] = None) -> Tuple[DS, DSOpt]:
        raise NotImplementedError()

    def get_classes(self) -> Collection[Union[int, str]]:
        raise NotImplementedError()

    def get_props(self) -> DatasetProps:
        raise NotImplementedError()

    def get_signal(self, amount: int) -> TTensor:
        raise NotImplementedError()

    def get_noise(self, amount: int) -> TTensor:
        raise NotImplementedError()
