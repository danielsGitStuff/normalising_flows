from __future__ import annotations
import math
import shutil
from pathlib import Path
from typing import Optional, Tuple, Collection, Union, List

import numpy as np

from common import jsonloader
from common.NotProvided import NotProvided
from common.globals import Global
from common.jsonloader import Ser
from distributions.Distribution import Distribution
from distributions.base import TTensor, cast_dataset_to_tensor, TTensorOpt, cast_to_ndarray
from maf.DS import DataLoader, DatasetProps, DS, DSOpt
from maf.stuff.StaticMethods import StaticMethods


class DataSource(Ser):
    """Unites multiple ways of obtaining data: tf.Dataset, path to a tf.Dataset, distributions.Distribution or another DataSource"""

    def __init__(self, ds_path: Optional[Path] = NotProvided(), distribution: Optional[Distribution] = NotProvided(), ds: DS = NotProvided(),
                 data_source: Optional[DataSource] = NotProvided()):
        super().__init__()
        self.ds: Optional[DS] = NotProvided.value_if_not_provided(ds, None)
        self.data_source: Optional[DataSource] = NotProvided.value_if_not_provided(data_source, None)
        self.ds_path: Optional[Path] = ds_path
        self.distribution: Optional[Distribution] = NotProvided.value_if_not_provided(distribution, None)
        self.ignored.add('ds')

    def ref(self) -> DataSource:
        """@return a reference to this instance."""
        src = DataSource(data_source=self)
        return src

    def get_data(self, amount: int) -> DS:
        def check(d: DS) -> DS:
            if len(d) < amount:
                raise ValueError(f"Length is {len(d)}, but {amount} is requested")
            return d.take(amount)

        if NotProvided.is_provided(self.data_source) and self.data_source is not None:
            return self.data_source.get_data(amount=amount)
        if NotProvided.is_provided(self.ds) and self.ds is not None:
            return check(self.ds)
        import tensorflow as tf

        if self.ds_path is not None and self.ds_path.exists():
            ds: DS = tf.data.experimental.load(str(self.ds_path))
            return check(ds)
        data = self.distribution.sample(amount, batch_size=Global.get_default('datasource_batch_size', 1000))
        return DS.from_tensor_slices(data)


class DL2(Ser):
    """Represents a dataset consisting of two tf.Dataset. One for signals, one for noise. Stores them in one folder, with properties file in JSON format.
    You can create a new DL2 handing it two DataSource objects, the amounts to grab of each and a directory in the constructor and call 'create_data()'."""

    @staticmethod
    def can_load(dir: Path) -> bool:
        js_file = Path(dir, 'dl2.json')
        if js_file.exists():
            return True
        return False

    @staticmethod
    def load_props_impl(dir: Path) -> DatasetProps:
        """do not call directly, use DL2.load_props() instead."""
        dl = DL2.load(dir)
        return dl.props

    @staticmethod
    def load_props(dir: Path) -> DatasetProps:
        # todo check deprecation
        return Global.POOL().run_blocking(DL2.load_props_impl, args=(dir,))

    @staticmethod
    def load(dir: Path) -> DL2:
        js_file = Path(dir, 'dl2.json')
        return jsonloader.load_json(js_file, raise_on_404=True)

    @staticmethod
    def execute_static_create_data(js: str, normalise: bool = False) -> str:
        dl: DL2 = jsonloader.from_json(js)
        dl.create_data(normalise=normalise)
        js_result = jsonloader.to_json(dl, pretty_print=True)
        return js_result

    def __init__(self, dataset_name: str = NotProvided(),
                 dir: Path = NotProvided(),
                 signal_source: DataSource = NotProvided(),
                 noise_source: DataSource = NotProvided(),
                 amount_of_signals: int = NotProvided(),
                 amount_of_noise: int = NotProvided()):
        super().__init__()
        self.amount_of_signals: int = amount_of_signals
        self.amount_of_noise: int = amount_of_noise
        self.dataset_name: str = dataset_name  # todo obsolete
        self.dir: Path = NotProvided.value_if_not_provided(dir, NotProvided.value_if_provided(dataset_name, lambda name: Path(StaticMethods.cache_dir(), name)))
        self.signal_dir: Path = NotProvided.value_if_provided(dataset_name, lambda name: Path(self.dir, "signal"))
        self.noise_dir: Path = NotProvided.value_if_provided(dataset_name, lambda name: Path(self.dir, "noise"))
        signal_source.ds_path = self.signal_dir
        noise_source.ds_path = self.noise_dir
        self.js_file: Path = NotProvided.value_if_provided(dataset_name, lambda _: Path(self.dir, 'dl2.json'))
        self.props_file: Path = NotProvided.value_if_provided(dataset_name, lambda _: Path(self.dir, 'props.json'))
        self.signal_source: DataSource = signal_source
        self.noise_source: DataSource = noise_source
        self.conditional_dims: int = 1
        self.props: Optional[DatasetProps] = None

    def clone(self, dir: Path) -> DL2:
        dl = DL2(dataset_name='dlnameewrew',
                 dir=dir,
                 signal_source=self.signal_source.ref(),
                 noise_source=self.noise_source.ref(),
                 amount_of_noise=self.amount_of_noise,
                 amount_of_signals=self.amount_of_signals)
        dl.create_data()
        return dl

    def split(self, test_dir: Path, test_split: float = None, test_amount: int = None) -> DL2:
        """BUG: when overwriting the current source directory of a tf.Dataset, the new one will be empty
        @note this code loads the entire dataset into the memory to work around that bug.
        """
        print(f"splitting dataset '{self.dir}'. Split goes to '{test_dir}'")
        signal = self.signal_source.get_data(self.props.no_of_signals)
        noise = self.noise_source.get_data(self.props.no_of_noise)
        signal, _ = cast_dataset_to_tensor(signal)
        noise, _ = cast_dataset_to_tensor(noise)
        if test_split is None and test_amount is None:
            raise RuntimeError("too much None")
        if test_split is not None and test_amount is not None:
            raise RuntimeError("too much not None")
        if test_split is not None:
            take_test_noise: int = math.floor(test_split * self.props.no_of_noise)
            take_test_signals: int = math.floor(test_split * self.props.no_of_signals)
        else:
            signal_noise_ratio = len(signal) / (len(signal) + len(noise))
            take_test_signals: int = math.ceil(signal_noise_ratio * test_amount)
            take_test_noise: int = test_amount - take_test_signals
        rest_noise: int = self.props.no_of_noise - take_test_noise
        rest_signals: int = self.props.no_of_signals - take_test_signals
        test_signals = signal[:take_test_signals]
        test_noise = noise[:take_test_noise]
        new_signal = signal[take_test_signals:]
        new_noise = noise[take_test_noise:]
        self.props.no_of_signals = rest_signals
        self.props.no_of_noise = rest_noise
        self.props.length = rest_noise + rest_signals

        self.amount_of_noise = rest_noise
        self.amount_of_signals = rest_signals
        import tensorflow as tf
        test = DL2(dataset_name='test',
                   dir=test_dir,
                   signal_source=DataSource(ds=DS.from_tensor_slices(test_signals)),
                   noise_source=DataSource(ds=DS.from_tensor_slices(test_noise)),
                   amount_of_noise=take_test_noise,
                   amount_of_signals=take_test_signals)
        test.create_data()
        shutil.rmtree(self.signal_dir)
        shutil.rmtree(self.noise_dir)
        tf.data.experimental.save(DS.from_tensor_slices(new_signal), str(self.signal_dir))
        tf.data.experimental.save(DS.from_tensor_slices(new_noise), str(self.noise_dir))
        jsonloader.to_json(self, file=self.js_file, pretty_print=True)
        jsonloader.to_json(self.props, file=self.props_file, pretty_print=True)
        return test

    def split2(self, test_dir: Path, take_test_sig: int, take_test_noi: int) -> DL2:
        """BUG: when overwriting the current source directory of a tf.Dataset, the new one will be empty
        @note this code loads the entire dataset into the memory to work around that bug.
        """
        print(f"splitting dataset '{self.dir}'. Split goes to '{test_dir}'")
        signal = self.signal_source.get_data(self.props.no_of_signals)
        noise = self.noise_source.get_data(self.props.no_of_noise)
        signal, _ = cast_dataset_to_tensor(signal)
        noise, _ = cast_dataset_to_tensor(noise)

        rest_noise: int = self.props.no_of_noise - take_test_noi
        rest_signals: int = self.props.no_of_signals - take_test_sig
        test_signals = signal[:take_test_sig]
        test_noise = noise[:take_test_noi]
        new_signal = signal[take_test_sig:]
        new_noise = noise[take_test_noi:]
        self.props.no_of_signals = rest_signals
        self.props.no_of_noise = rest_noise
        self.props.length = rest_noise + rest_signals

        self.amount_of_noise = rest_noise
        self.amount_of_signals = rest_signals
        import tensorflow as tf
        test = DL2(dataset_name='test',
                   dir=test_dir,
                   signal_source=DataSource(ds=DS.from_tensor_slices(test_signals)),
                   noise_source=DataSource(ds=DS.from_tensor_slices(test_noise)),
                   amount_of_noise=take_test_noi,
                   amount_of_signals=take_test_sig)
        test.create_data()
        shutil.rmtree(self.signal_dir)
        shutil.rmtree(self.noise_dir)
        tf.data.experimental.save(DS.from_tensor_slices(new_signal), str(self.signal_dir))
        tf.data.experimental.save(DS.from_tensor_slices(new_noise), str(self.noise_dir))
        jsonloader.to_json(self, file=self.js_file, pretty_print=True)
        jsonloader.to_json(self.props, file=self.props_file, pretty_print=True)
        return test

    def create_data_in_process(self, normalise: bool = False) -> DL2:
        js = jsonloader.to_json(self)
        # print('debug skip pool')
        # return DL2.execute_static_create_data(js)
        js_result: str = Global.POOL().run_blocking(DL2.execute_static_create_data, args=(js, normalise))
        dl: DL2 = jsonloader.from_json(js_result)
        return dl

    def create_data(self, normalise: bool = False) -> DL2:
        """reads the specified amount of samples from noise and signal and writes them into 'dir'.
        Will not execute if an existing DL2 is present in 'dir'."""
        # todo run in own process
        amount_signal = self.amount_of_signals
        amount_noise = self.amount_of_noise
        if self.props_file.exists():
            self.props = jsonloader.load_json(self.props_file)
            return self
        print(f"creating data in '{self.dir}'")
        self.props = DatasetProps()
        self.props.length = amount_signal + amount_noise
        self.props.no_of_noise = amount_noise
        self.props.no_of_signals = amount_signal
        self.props.classes = [1, 0]
        self.props.conditional_dimensions = self.conditional_dims
        ds_signal = self.signal_source.get_data(amount=amount_signal)
        ds_noise = self.noise_source.get_data(amount=amount_noise)
        import tensorflow as tf
        if normalise:
            signal, _ = cast_dataset_to_tensor(ds_signal)
            noise, _ = cast_dataset_to_tensor(ds_noise)
            signal = cast_to_ndarray(signal)
            noise = cast_to_ndarray(noise)
            t = np.concatenate([signal, noise])
            epsilon = np.finfo(np.float32).eps
            std = t.std(axis=0)
            std = np.maximum(std, epsilon)
            mean = t.mean(axis=0)
            signal = (signal - mean) / std
            noise = (noise - mean) / std
            ds_signal = DS.from_tensor_slices(signal)
            ds_noise = DS.from_tensor_slices(noise)
        self.props.dimensions = ds_signal.element_spec.shape[0]
        tf.data.experimental.save(ds_signal, str(self.signal_dir))
        tf.data.experimental.save(ds_noise, str(self.noise_dir))
        self.signal_source = DataSource(ds_path=self.signal_dir)
        self.noise_source = DataSource(ds_path=self.noise_dir)
        jsonloader.to_json(self, file=self.js_file, pretty_print=True)
        jsonloader.to_json(self.props, file=self.props_file, pretty_print=True)
        return self

    def get_signal(self, amount: int) -> DS:
        import tensorflow as tf
        ds_signal: DS = tf.data.experimental.load(str(self.signal_dir))
        if len(ds_signal) < amount:
            raise ValueError(f"Requested {amount} signal samples, but only got {len(ds_signal)}")
        return ds_signal.take(amount)

    def get_noise(self, amount: int) -> DS:
        import tensorflow as tf
        ds_noise: DS = tf.data.experimental.load(str(self.noise_dir))
        if len(ds_noise) < amount:
            raise ValueError(f"Requested {amount} noise samples, but only got {len(ds_noise)}")
        return ds_noise.take(amount)

    def get_conditional(self, amount_signal: int = None, amount_noise: int = None) -> DS:
        if amount_signal is None:
            amount_signal = self.props.no_of_signals
        if amount_noise is None:
            amount_noise = self.props.no_of_noise
        signal = self.get_signal(amount=amount_signal)
        noise = self.get_noise(amount=amount_noise)
        import tensorflow as tf
        signal = signal.map(lambda t: tf.concat([tf.constant([1.0], dtype=tf.float32), t], axis=0))
        noise = noise.map(lambda t: tf.concat([tf.constant([0.0], dtype=tf.float32), t], axis=0))
        return signal.concatenate(noise)


class DL3(Ser):
    """produces a DL2 object when executed. This is useful in case you might want to download a CSV file from somewhere and preprocess it and then create a DL2 data set from
    there. """

    @staticmethod
    def execute_static(js: str) -> str:
        dl3: DL3 = jsonloader.from_json(js)
        dl2 = dl3.fetch()
        dl2_js = jsonloader.to_json(dl2, pretty_print=True)
        return dl2_js

    def __init__(self, url: str = NotProvided(), dl_folder: Path = NotProvided()):
        super().__init__()
        self.dl_folder: Path = dl_folder
        self.url: str = url
        self.amount_of_signals: Optional[int] = None
        self.amount_of_noise: Optional[int] = None

    def fetch(self) -> DL2:
        self.fetch_impl()
        return DL2.load(self.dl_folder)

    def fetch_impl(self):
        raise NotImplementedError(f"download your dataset from '{self.url}' and create a DL2 in '{self.dl_folder}' here. set no_of_signals etc as well")

    def execute(self) -> DL2:
        if DL2.can_load(self.dl_folder):
            dl2 = DL2.load(self.dl_folder)
        else:
            js = jsonloader.to_json(self)
            # print('debug skip pool')
            # dl2_js = DL3.execute_static(js)
            # return jsonloader.from_json(dl2_js)
            dl2_js = Global.POOL().run_blocking(DL3.execute_static, args=(js,))
            dl2: DL2 = jsonloader.from_json(dl2_js)
        return dl2


if __name__ == '__main__':
    dl = DL2.load('/home/xor/Documents/normalising_flows/.cache/mixlearn_miniboone_default/dl_train/')
    dl.get_signal(40000)
    print('ende')
