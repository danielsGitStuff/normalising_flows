import math
import sys
from typing import List, Callable

import numpy as np
import tensorflow as tf
from keras.utils.data_utils import Sequence

from common.util import Runtime
from distributions.base import enable_memory_growth
from maf.DS import DS


class KLSampler(Sequence):
    def __init__(self, dims: int, half_width: float, step_size: float, batch_size: int):
        self.half_width: float = half_width
        self.step_size: float = step_size
        self.dims: int = dims
        self.batch_size: int = batch_size
        self.xs: List[float] = []
        self.batch_sizes: List[int] = []
        v: float = -self.half_width
        while v <= self.half_width:
            self.xs.append(v)
            v += self.step_size

        self.possible_combinations: int = int(math.pow(len(self.xs), self.dims))
        left: float = self.possible_combinations
        while left > 0:
            take = math.floor(min(self.batch_size, left))
            self.batch_sizes.append(take)
            left -= take

        self.steps: int = len(self.xs)
        self.potencies: List[int] = [math.floor(math.pow(self.steps, i)) for i in range(self.dims)]
        self.potencies = list(reversed(self.potencies))

    def __len__(self) -> int:
        return len(self.batch_sizes)

    def __getitem__(self, index):
        """@return a batch of samples"""
        batch_size = self.batch_sizes[index]
        product_index = self.batch_size * index

        # product_index = 543
        indices = [0] * self.dims
        rest = product_index
        for i, p in enumerate(self.potencies):
            if product_index < p:
                continue
            v = math.floor(rest / p)
            rest = rest - (v * p)
            indices[i] = v
        batch = []
        # INVERT indices, so a lot of reversed() stuff and indices can be averted
        indices = list(reversed(indices))
        for _ in range(batch_size):
            xs = [self.xs[i] for i in indices]
            # try:
            #     xs = [self.xs[i] for i in indices]
            # except Exception:
            #     print("fail")
            batch.append(xs)
            overhead: int = 1
            for i, index in enumerate(indices):
                new = index + overhead
                if new == self.steps:
                    indices[i] = 0
                    overhead = 1
                else:
                    indices[i] = new
                    break

        batch = tf.constant(batch, dtype=tf.float32)
        return batch

    def to_gen(self) -> Callable:
        def f():
            return self

        return f

    def to_dataset(self) -> DS:
        return DS.from_generator(self.to_gen(), output_signature=tf.TensorSpec(shape=(None, self.dims), dtype=tf.float32))


class StatefulKLSampler:
    def __init__(self, dims: int, half_width: float, step_size: float):
        for _ in range(5):
            print('                    THIS (StatefulKLSampler) IMPLEMENTATION IS AROUND 30X SLOWER THAN KLSampler!', file=sys.stderr)
        self.half_width: float = half_width
        self.step_size: float = step_size
        self.dims: int = dims
        self.wheel: List[float] = []
        v: float = -self.half_width
        while v <= self.half_width:
            self.wheel.append(v)
            v += self.step_size

        self.possible_combinations: int = int(math.pow(len(self.wheel), self.dims))
        self.steps: int = len(self.wheel)
        self.potencies: List[int] = [math.floor(math.pow(self.steps, i)) for i in range(self.dims)]
        self.potencies = list(reversed(self.potencies))
        self.indices = [0] * self.dims

    def __to_generator_function__(self):
        def f():
            """@return a single sample"""
            done = 0
            combinations = self.possible_combinations
            wheel = self.wheel
            while done < combinations:
                xs = [wheel[i] for i in self.indices]
                yield xs
                overhead: int = 1
                for i, index in enumerate(self.indices):
                    new = index + overhead
                    if new == self.steps:
                        self.indices[i] = 0
                        overhead = 1
                    else:
                        self.indices[i] = new
                        break
                done += 1

        return f

    def to_dataset(self) -> DS:
        return DS.from_generator(self.__to_generator_function__(), output_signature=tf.TensorSpec(shape=(self.dims,), dtype=tf.float32))


if __name__ == '__main__':
    enable_memory_growth()
    BS: int = 100000
    stateful_kl_sampler = StatefulKLSampler(dims=4, half_width=1.0, step_size=.01)
    kl_sampler = KLSampler(dims=4, half_width=1.0, step_size=.01, batch_size=BS)


    def benchmark(ds: DS):
        amount = 5
        print('starting benchmark')
        r = Runtime(f"measuring creation of {amount} batches").start()
        for i, batch in enumerate(ds):
            print(f"batch {i} shape is {batch.shape}")
            if i == amount - 1:
                break
        r.stop().print()


    benchmark(stateful_kl_sampler.to_dataset().batch(BS, num_parallel_calls=tf.data.AUTOTUNE))
    benchmark(kl_sampler.to_dataset())
