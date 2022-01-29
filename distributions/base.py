from __future__ import annotations

import sys

from common.globals import Global
from typing import Union, Optional, Tuple, Dict, List, Callable
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.data import Dataset
from tensorflow.python.framework.ops import EagerTensor

# types
from tensorflow.python.ops.gen_dataset_ops import BatchDataset

TData = Union[Tensor, np.ndarray, tf.data.Dataset, float]
TTensor = Union[Tensor, np.ndarray, List[float], List[List[float]], Dataset]
MaybeConditional = Union[TTensor, Tuple[TTensor, TTensor]]
MaybeBijKwargs = Optional[Dict[str, Union[Dict[str, Tensor], Dict[str, Dict]]]]
TTensorOpt = Optional[TTensor]
TDataOpt = Optional[TData]


class BaseMethods:
    @staticmethod
    def extract_xs_cond(xs: Union[TTensor, Tuple[Tensor, Tensor]], cond: TTensorOpt = None) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(xs, Tuple) and cond is not None:
            raise ValueError("Conditional was provided via 'cond' and 'xs'")
        if cond is not None:
            x, _ = cast_to_tensor(xs)
            c, _ = cast_to_tensor(cond)
            return x, c
        return cast_to_tensor(xs)
        # if isinstance(xs, Tuple):
        #     return cast_to_tensor(xs[0]), cast_to_tensor(xs[1])
        # xs: TTensor = xs
        # if cond is None:
        #     return cast_to_tensor(xs), None
        # else:
        #     return cast_to_tensor(xs), cast_to_tensor(cond)

    @staticmethod
    def is_conditional_Dataset(ds: tf.data.Dataset):
        if isinstance(ds.element_spec, Tuple):
            return True
        return False

    @staticmethod
    def is_conditional_data(data: TData):
        if isinstance(data, tf.data.Dataset):
            return BaseMethods.is_conditional_Dataset(data)
        elif isinstance(data, Tuple) and len(data) == 2:
            return True
        return False

    @staticmethod
    def random_covariance_matrix(n: int, sample_f: Callable[[], float], dtype=np.float32) -> np.ndarray:
        m = np.zeros((n, n), dtype=dtype)
        for i in range(n):
            j = 0
            for j in range(n):
                if j > i:
                    break
                cov = sample_f()
                if i == j:
                    cov = np.abs(cov)
                m[i, j] = cov
                m[j, i] = cov
        # print(m)
        return m

    @staticmethod
    def un_nan(t: Tensor, replacement=0.0) -> Tensor:
        nans = tf.math.is_nan(t)
        t = tf.where(tf.logical_not(nans), t, replacement)
        return t


def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    gpu_index = Global.get_default('tf_gpu', 0)
    tf.config.set_visible_devices([gpus[gpu_index]], 'GPU')
    # tf.config.set_visible_devices()
    # if gpus:
    #     # Create 2 virtual GPUs with 1GB memory each
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             gpus[0],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
    #              tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)


def enable_memory_growth():
    print('enabling memory growth ...')
    set_gpu()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        raise RuntimeError('no physical GPUs available!!!')
    try:
        for card in physical_devices:
            tf.config.experimental.set_memory_growth(card, True)
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        print('could not enable cuda memory growth', file=sys.stderr)
        print('physical devices:', file=sys.stderr)
        for d in physical_devices:
            print(f"   {d}", file=sys.stderr)


def cast_to_ndarray(tensor: Union[Tensor, np.ndarray], dtype=np.float32) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == dtype:
            return tensor
        return tensor.astype(dtype)
    if isinstance(tensor, EagerTensor):
        n: np.ndarray = tensor.numpy()
        if n.dtype == dtype:
            return n
        return n.astype(dtype)
    if isinstance(tensor, np.float):
        return np.array(tensor, dtype=dtype)
    a = tf.make_ndarray(tf.cast(tensor, dtype=tf.float32))
    return a.astype(dtype)


def cast_to_tensor(data: TData, dtype=tf.float32) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    if isinstance(data, tf.data.Dataset):
        return cast_dataset_to_tensor(data)
    if isinstance(data, tf.Tensor):
        if data.dtype == dtype:
            return data, None
        return tf.cast(data, dtype), None
    if isinstance(data, Tuple) and len(data) == 2:
        return tf.convert_to_tensor(data[0], dtype=dtype), tf.convert_to_tensor(data[1], dtype=dtype)
    tensor = tf.convert_to_tensor(data, dtype=dtype)
    return tensor, None


def cast_dataset_to_tensor(dataset: tf.data.Dataset) -> Tuple[Tensor, Optional[Tensor]]:
    if BaseMethods.is_conditional_Dataset(dataset):
        if len(dataset.element_spec[0].shape) < 2:
            dataset = dataset.batch(batch_size=10000)
        content = []
        conds = []
        for b, c in dataset:
            content.append(b)
            conds.append(c)
        xs = tf.concat(content, axis=0)
        cond = tf.concat(conds, axis=0)
        xs, _ = cast_to_tensor(xs)
        cond, _ = cast_to_tensor(cond)
        return xs, cond

    if len(dataset.element_spec.shape) < 2:
        dataset = dataset.batch(batch_size=10000)
        # dataset = dataset.batch(10)

    if len(dataset.element_spec.shape) >= 1:  # is batched
        content = []
        for b in dataset:
            content.append(b)
        if len(dataset.element_spec.shape) == 1 and dataset.element_spec.shape[0] is not None:
            result = tf.stack(content)
        else:
            result = tf.concat(content, axis=0)
        return cast_to_tensor(result)
    raise RuntimeError('something went wrong with datset casting')
