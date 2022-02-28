import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.python.data import Dataset

from common.util import Runtime
from common.NotProvided import NotProvided
from distributions.base import enable_memory_growth
from broken.DS import DataLoader
from maf.mixlearn.MixLearnExperiment import MixLearnExperiment, DatasetFetcher
from maf.stuff.StaticMethods import StaticMethods
import tensorflow as tf


class HepMassDataLoader(DataLoader):
    """This file is likely not compatible with the rest of the framework since my work focused on MiniBooNE instead"""

    def __init__(self, norm_data: bool, conditional: bool = False):
        print("This file is likely not compatible with the rest of the framework since my work focused on MiniBooNE instead", file=sys.stderr)
        sys.exit(2)
        super().__init__(conditional=conditional)
        self.norm_data: bool = norm_data
        self.cache_txt_train_file: Path = Path(StaticMethods.cache_dir(), 'hepmass_train.txt')
        self.cache_txt_test_file: Path = Path(StaticMethods.cache_dir(), 'hepmass_test.txt')
        self.cache_np_train_file: Path = Path(StaticMethods.cache_dir(), 'hepmass_train.npy')
        self.cache_np_test_file: Path = Path(StaticMethods.cache_dir(), 'hepmass_test.npy')
        self.dataset_train_url: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_train.csv.gz'
        self.dataset_test_url: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_test.csv.gz'
        self.dev_cpu = tf.config.list_logical_devices('CPU')[0]
        self.dev_gpu = tf.config.list_logical_devices('GPU')[0]
        self.val_ratio: float = 0.1

    def create_data_sets(self, batch_size: Optional[int] = NotProvided()) -> Tuple[Dataset, Optional[Dataset]]:
        def load(dataset_train_url: str, cache_np_file: Path) -> tf.data.Dataset:
            if cache_np_file.exists():
                print(f"loading cached dataset '{cache_np_file}'")
                r = Runtime("load cache file").start()
                data = np.load(cache_np_file, allow_pickle=True)
                r.stop().print()
                with tf.device(self.dev_cpu.name):
                    return tf.data.Dataset.from_tensor_slices(data)
            else:
                cache_txt_file = cache_np_file.with_name(f"{cache_np_file.name}.txt")
                r = Runtime(f"fetch '{dataset_train_url}' -> '{cache_txt_file}'").start()
                train_fetcher = DatasetFetcher(dataset_url=dataset_train_url, target_file=cache_txt_file, extract='gz')
                train_fetcher.fetch()
                r.stop().print()
                r = Runtime(f"read '{cache_txt_file}'").start()
                columns = ['label'] + [f"f{i}" for i in range(27)]
                df = pd.read_csv(cache_txt_file, header=0, names=columns, dtype=np.float32)
                if self.norm_data:
                    df = (df - df.mean()) / df.std()
                r.stop().print()
                print(df.head())
                r = Runtime(f"store '{cache_txt_file}' -> '{cache_np_file}'").start()
                np.save(cache_np_file, df.values)
                r.stop().print()
                with tf.device(self.dev_cpu.name):
                    return tf.data.Dataset.from_tensor_slices(df.values)

        # test_ds = load(self.dataset_test_url, self.cache_np_test_file)
        train_ds = load(self.dataset_train_url, self.cache_np_train_file)
        # train_ds = train_ds.shuffle(buffer_size=len(train_ds))
        val: int = math.ceil(len(train_ds) * self.val_ratio)
        train_val_ds = train_ds.take(val)
        train_ds = train_ds.skip(val)
        batch_size = NotProvided.value_if_not_provided(batch_size, 128)

        def prepare(d: tf.data.Dataset, cache_file: Path) -> tf.data.Dataset:
            # d = d.cache(filename=str(cache_file))
            return d

        train_ds = prepare(train_ds, Path(StaticMethods.cache_dir(), 'hepmass_tf_train'))
        train_val_ds = prepare(train_val_ds, Path(StaticMethods.cache_dir(), 'hepmass_tf_val'))
        return train_ds, train_val_ds


class MixLearnExperimentHepMass(MixLearnExperiment):
    def __init__(self, name: str,
                 epochs: int,
                 layers: int,
                 batch_size: Optional[int] = None,
                 data_batch_size: Optional[int] = 128,
                 hidden_shape: List[int] = [200, 200],
                 norm_layer: bool = False,
                 norm_data: bool = True,
                 noise_variance: float = 0.0,
                 batch_norm: bool = False,
                 use_tanh_made: bool = False, conditional: bool = False):
        super().__init__(name=name, clf_epochs=epochs, layers=layers, batch_size=batch_size, data_batch_size=data_batch_size, hidden_shape=hidden_shape, norm_layer=norm_layer,
                         norm_data=norm_data, noise_variance=noise_variance,
                         batch_norm=batch_norm, use_tanh_made=use_tanh_made, conditional_dims=conditional)

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(StaticMethods.cache_dir(), "hepmass_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def create_data_loader(self, norm_data: bool) -> DataLoader:
        loader = HepMassDataLoader(conditional=self.conditional_dims, norm_data=norm_data)
        return loader


if __name__ == '__main__':
    enable_memory_growth()
    tf.random.set_seed(1234)
    h = MixLearnExperimentHepMass(name="hm_test", epochs=10,
                                  batch_size=10000,
                                  data_batch_size=None,
                                  layers=5,
                                  hidden_shape=[512, 512],
                                  norm_layer=False,
                                  norm_data=True,
                                  noise_variance=0.0,
                                  batch_norm=True,
                                  use_tanh_made=True)
