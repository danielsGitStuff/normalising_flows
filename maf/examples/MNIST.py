from __future__ import annotations

import math
import os
import sys

from common.globals import Global
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import Tensor
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from common.util import Runtime
from distributions.LearnedDistribution import LearnedDistribution, EarlyStop
from distributions.base import cast_to_ndarray, enable_memory_growth
from maf.ClassOneHot import ClassOneHot
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.DS import DS, DataLoader
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from maf.examples.stuff.StaticMethods import StaticMethods


class MnistLoader(DataLoader):
    class Methods:
        @staticmethod
        def filter_numbers(numbers: List[int]):
            numbers = list(set(numbers))
            numbers: Tensor = tf.convert_to_tensor(numbers, dtype=tf.int64)
            n = len(numbers)

            def f(x, y):
                y = tf.repeat(y, n)
                return tf.reduce_any(tf.equal(y, numbers))
                # return tf.math.equal(y, number)

            return f

    def __init__(self, numbers: Optional[List[int]] = None, conditional: bool = False, norm: Optional[str] = None, limit: Optional[int] = None,
                 val_limit: Optional[int] = None,
                 cache: bool = False, noise_variance: float = 0.0):
        super().__init__(conditional=conditional)
        self.numbers: Optional[List[int]] = numbers
        self.norm: Optional[str] = norm
        self.limit: Optional[int] = limit
        self.val_limit: Optional[int] = val_limit
        self.cache: bool = cache
        self.noise_variance: float = noise_variance

    def create_data_sets(self) -> Tuple[Dataset, Optional[Dataset]]:
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=False,
            as_supervised=True,
            with_info=True,
        )
        print(f"loaded {len(ds_train)} training samples")
        if self.numbers is not None:
            ds_train: DatasetV2 = ds_train.filter(MnistLoader.Methods.filter_numbers(self.numbers))
            ds_test: DatasetV2 = ds_test.filter(MnistLoader.Methods.filter_numbers(self.numbers))
        if self.norm is not None:
            if self.norm == "-1 1":
                ds_train = ds_train.map(
                    DataLoader.Methods.normalize_img_minus1_1(), num_parallel_calls=tf.data.AUTOTUNE)
                ds_test = ds_test.map(
                    DataLoader.Methods.normalize_img_minus1_1(), num_parallel_calls=tf.data.AUTOTUNE)
            elif self.norm == "0 1":
                ds_train = ds_train.map(
                    DataLoader.Methods.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
                ds_test = ds_test.map(
                    DataLoader.Methods.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            elif self.norm == "logit":
                pass  # will norm in the next step
            else:
                raise RuntimeError(f"unknown value for norm: {self.norm}")

        def _to_dataset(ds: DatasetV2) -> tf.data.Dataset:
            ts: List[Tensor] = []
            ls: List[Tensor] = []
            iter = tfds.as_numpy(ds)
            norm_f: Callable[[Tensor], Tensor] = lambda xs: xs
            if self.norm == 'logit': norm_f = lambda xs: DataLoader.Methods.logit_xs(xs)
            if self.conditional:
                for tup in iter:
                    t: Tensor = tf.convert_to_tensor(tup[0], dtype=tf.float32)
                    l: Tensor = tf.convert_to_tensor(tup[1], dtype=tf.float32)
                    t = tf.reshape(t, [-1])
                    t = norm_f(t)
                    ts.append(t)
                    ls.append(l)
            else:
                for tup in iter:
                    t: Tensor = tf.convert_to_tensor(tup[0], dtype=tf.float32)
                    t = tf.reshape(t, [-1])  # flatten
                    t = norm_f(t)
                    ts.append(t)
            if self.conditional:
                tss = tf.convert_to_tensor(ts)
                lss = tf.convert_to_tensor(ls)
                lss = tf.reshape(lss, (len(ls), 1))
                return DS.from_tensor_slices(tf.concat([lss, tss], axis=1))
                # return tf.data.Dataset.from_tensor_slices((tss, lss))
            return tf.data.Dataset.from_tensor_slices(ts)

        rt = Runtime("convert datasets").start()
        t_train: tf.data.Dataset = _to_dataset(ds_train)
        t_test: tf.data.Dataset = _to_dataset(ds_test)

        rt.stop().print()
        print(f"{len(t_train)} training samples left")
        if self.cache:
            t_train = t_train.cache()
            t_test = t_test.cache()
        t_train = t_train.shuffle(buffer_size=len(t_train))
        t_test = t_test.shuffle(buffer_size=len(t_test))
        if self.limit is not None:
            t_train = t_train.take(self.limit)
        if self.val_limit is not None:
            t_test = t_test.take(self.val_limit)

        if self.noise_variance > 0.0:
            print(f"adding gaussian noise with variance {self.noise_variance} to training data")
            if self.conditional:
                f: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = lambda t, cond: (t + tf.random.normal(t.shape, stddev=self.noise_variance), cond)
            else:
                f: Callable[[Tensor], Tensor] = lambda t: t + tf.random.normal(t.shape, stddev=self.noise_variance)
            t_train = t_train.map(f)
        # batch_size = NotProvided.value_if_not_provided(batch_size, self.batch_size)
        # if batch_size is not None:
        #     t_train = t_train.batch(batch_size, drop_remainder=True)
        #     t_test = t_test.batch(batch_size, drop_remainder=True)
        return t_train, t_test


class Mist:
    def __init__(self, numbers: Optional[Union[int, List[int]]] = None, norm_data: Optional[str] = "logit", norm_layer: bool = False, batch_norm: bool = True, epochs: int = 50,
                 use_tanh_made: bool = True, layers: int = 8, hidden_shape: List[int] = [256, 256], conditional: bool = False, one_hot: bool = False, activation: str = "relu",
                 dataset_noise_variance: float = 0.0, noise_layer_variance: float = 0.0):
        self.numbers: Optional[List[int]] = numbers if isinstance(numbers, List) else \
            None if numbers is None else [numbers]
        self.noise_variance: float = dataset_noise_variance
        self.conditional: bool = conditional
        self.conditional_dim: int = 1 if self.conditional else 0
        self.one_hot: bool = one_hot
        self.noise_layer_variance: float = noise_layer_variance
        self.activation: str = activation
        self.maf: Optional[MaskedAutoregressiveFlow] = None
        if self.conditional:
            numbers = self.numbers if self.numbers is not None else list(range(0, 10))
            class_one_hot = ClassOneHot(enabled=False)
            if self.one_hot:
                print("using ONE HOT encoding for class")
                class_one_hot = ClassOneHot(enabled=True, num_tokens=len(numbers), classes=numbers, typ='int').init()
            self.maf = MaskedAutoregressiveFlow(input_dim=28 * 28, layers=layers, batch_norm=batch_norm, hidden_shape=hidden_shape, norm_layer=norm_layer,
                                                use_tanh_made=use_tanh_made,
                                                conditional_dims=self.conditional_dim, class_one_hot=class_one_hot, activation=self.activation,
                                                input_noise_variance=noise_layer_variance)
        else:
            self.maf = MaskedAutoregressiveFlow(input_dim=28 * 28, layers=layers, batch_norm=batch_norm, hidden_shape=hidden_shape, norm_layer=norm_layer,
                                                use_tanh_made=use_tanh_made, input_noise_variance=noise_layer_variance, activation=self.activation)
        self.norm_data: Optional[str] = norm_data
        self.epochs: int = epochs

    def fit(self):
        # t_test: Dataset = tf.data.Dataset.from_tensors(t_test)
        dataloader = MnistLoader(numbers=self.numbers, conditional=self.conditional, norm=self.norm_data, limit=None, val_limit=1000, noise_variance=self.noise_variance)
        es: Optional[EarlyStop] = None
        es = EarlyStop(monitor="val_loss", comparison_op=tf.less_equal, patience=10, restore_best_model=True)
        self.maf.fit(dataset=dataloader, epochs=self.epochs, batch_size=128, early_stop=es)

    def test(self):
        DEBUG = 0
        # +1 is for the zero sample
        no_samples = 5
        no_digits = len(self.numbers)
        m: MaskedAutoregressiveFlow = self.maf
        results_dir: Path = Global.get_default('resuts_dir', Path('results_mnist'))
        results_dir.mkdir(exist_ok=True)
        print(f"sampling {no_samples + 1} images for each of the {no_digits} numbers")
        fig, axs = plt.subplots(no_digits, no_samples + 1)
        for ax in axs.flatten():
            ax.set_axis_off()
        axs = axs.reshape((no_digits, no_samples + 1))
        file_name = 'maf '
        if self.conditional:
            file_name = f"{file_name}cond "
        file_name = f"{file_name} numbers {self.numbers} l {m.layers}, e {self.epochs}, h {m.hidden_shape}, bn {m.batch_norm}, nd {self.norm_data}, nl {m.norm_layer}, tanh {m.use_tanh_made}.png"
        for number_index, number in enumerate(self.numbers):
            r = Runtime(f"sampling no '{number}' {no_samples + 1} times").start()
            print(r.name)
            s = 'maf '
            if self.conditional:
                s = f"{s} cond"
            s = f"{s} num {number} l {m.layers}, e {self.epochs}, h {m.hidden_shape}, bn {m.batch_norm}, nd {self.norm_data}, nl {m.norm_layer}, tanh {m.use_tanh_made}"

            zeros: np.ndarray = np.zeros(28 * 28, dtype=np.float32).reshape((1, 28 * 28))
            non_zeros: np.ndarray = np.random.normal(size=no_samples * 28 * 28) * 1.0
            non_zeros = non_zeros.reshape((no_samples, 28 * 28)).astype(np.float32)
            ys: np.ndarray = np.concatenate([zeros, non_zeros])
            cond = None
            if self.conditional:
                cond = [number] * (no_samples + 1)
            samples = m.calculate_xs(ys, cond)
            samples = cast_to_ndarray(samples)
            samples = samples.reshape(no_samples + 1, 28, 28)
            r.stop().print()
            if self.norm_data == "logit":
                samples = DataLoader.Methods.inverse_logit(samples)
            for sample_index, image in enumerate(samples):
                xs = samples[sample_index]
                # xs[0,DEBUG] = 4.5
                DEBUG += 1
                ax = axs[number_index, sample_index]
                ax.imshow(xs, cmap='gray')
        plt.tight_layout()
        plt.savefig(Path(results_dir, file_name))
        print('das')
        return

        s = "maf "
        if self.conditional:
            s = f"maf.cond "
        s += f"num {self.numbers}, l {m.layers},e {self.epochs}, h {m.hidden_shape}, bn {m.batch_norm}, logit {self.norm_data}, nl {m.norm_layer}, tanh {m.use_tanh_made}"

        print(f"sampling {no_samples} images")
        r = Runtime("sampling").start()

        zeros = np.zeros(28 * 28, dtype=np.float32).reshape((1, 28 * 28))
        cond = None
        if self.conditional:
            cond = np.random.choice(self.numbers, 1)
        from_zeros = self.maf.calculate_xs(zeros, cond)
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(cast_to_ndarray(from_zeros).reshape((28, 28)), cmap='gray')
        results_dir: Path = Global.get_default('resuts_dir', Path('results_mnist'))
        results_dir.mkdir(exist_ok=True)
        if self.conditional:
            plt.savefig(Path(results_dir, f"{s}.ZEROS.cond_{cond}.png"))
        else:
            plt.savefig(Path(results_dir, f"{s}.ZEROS.png"))
        title = ""
        if True:
            ys: np.ndarray = np.random.normal(size=no_samples * 28 * 28) * 1.0
            ys = ys.reshape((no_samples, 28 * 28)).astype(np.float32)
            cond = None
            if self.conditional:
                cond = np.random.choice(self.numbers, no_samples).reshape((no_samples, 1))
                title = ",".join([str(n) for n in cond])
            # sample = self.maf.transformed_distribution.bijector.forward(ys)
            sample = self.maf.calculate_xs(ys, cond)
            sample = cast_to_ndarray(sample)
            # sample = np.clip(sample, 0.0, 255.0)
            print(f"sample min  {sample.min()}")
            print(f"sample max  {sample.max()}")
            print(f"sample mean {sample.mean()}")
            print(f"sample var  {sample.var()}")

        else:
            sample = self.maf.sample(no_samples)

        r.stop().print()
        osample: np.ndarray = sample.reshape((no_samples, 28, 28))
        print(f"osample min  {osample.min()}")
        print(f"osample max  {osample.max()}")
        print(f"osample mean {osample.mean()}")
        print(f"osample var  {osample.var()}")

        sample = osample
        if self.norm_data == "logit":
            sample = DataLoader.Methods.inverse_logit(sample)
        print(f"denormalised sample min {tf.reduce_min(sample).numpy()}")
        print(f"denormalised sample max {tf.reduce_max(sample).numpy()}")

        sample = osample
        if no_samples == 1:
            plt.suptitle(title)
            # plt.title(title)
            plt.imshow(sample[0], cmap='gray')
        else:
            fig, axs = plt.subplots(int(math.ceil(no_samples / 3)), 3, figsize=(3 * 7, int(math.ceil(no_samples / 3)) * 7))
            axs: np.ndarray = axs
            plt.suptitle(title)
            # plt.title(title)
            axs = axs.flatten()
            for ax in axs:
                ax.set_axis_off()
            for i in range(no_samples):
                # for i, ax in enumerate(axs.flatten()):
                xs = sample[i]
                axs[i].imshow(xs, cmap='gray')
            # fig, axs = plt.subplots(no_samples, 1, figsize=(2, 2 * no_samples))
            # plt.suptitle(title)
            # # plt.title(title)
            # for i, ax in enumerate(axs):
            #     xs = sample[i]
            #     ax.imshow(xs, cmap='gray')
        plt.tight_layout()
        plt.savefig(Path(results_dir, f"{s}.png"))
