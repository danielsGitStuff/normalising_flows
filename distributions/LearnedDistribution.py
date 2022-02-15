from __future__ import annotations

import os
from abc import ABC
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Callable, Any

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from common.jsonloader import Ser
from common.NotProvided import NotProvided
from distributions.Distribution import Distribution, DensityPlotData
from distributions.base import TData, TTensorOpt, cast_dataset_to_tensor, TDataOpt
from maf.DS import DS, DSOpt, DataLoader
from maf.SaveSettings import SaveSettings


class LearnedConfig(Ser):
    """hold all the weights, learned parameters, randomly set permutations etc regarding a LearnedDistribution instance"""

    def __init__(self):
        super().__init__()

    def create(self) -> LearnedDistribution:
        """builds an entire new instance from the stored values"""
        raise NotImplementedError()


class LearnedDistribution(Distribution, ABC):
    class Methods:
        @staticmethod
        def cast_data(data: Optional[np.ndarray, Tensor]) -> Optional[tf.data.Dataset]:
            if data is None:
                return None
            if isinstance(data, tf.data.Dataset):
                return data
            tensor = tf.convert_to_tensor(data, dtype=tf.float32)
            return tf.data.Dataset.from_tensor_slices(tensor)

        @staticmethod
        def cast_tensor(tensor: Optional[Union[np.ndarray, Tensor]]) -> Optional[Tensor]:
            if tensor is None:
                return None
            if isinstance(tensor, Tensor):
                if tensor.dtype == tf.float32:
                    return tensor
                return tf.cast(tensor, dtype=tf.float32)
            return tf.convert_to_tensor(tensor, dtype=tf.float32)

    @staticmethod
    def can_load_from(folder: Union[Path, str], prefix: str) -> bool:
        complete_prefix = str(Path(folder, prefix))
        json_file = f"{complete_prefix}.model.json"
        checkpoint_file = f"{complete_prefix}.data-00000-of-00001"
        return os.path.exists(json_file) and os.path.exists(checkpoint_file)

    @staticmethod
    def load(folder: Union[Path, str], prefix: str) -> LearnedDistribution:
        raise NotImplementedError()

    def __init__(self, input_dim: int, conditional_dims: int):
        super().__init__(input_dim, conditional_dims=conditional_dims)
        self.history: FitHistory = FitHistory()
        self.ignored.add("history")

    def fit(self, dataset: DS,
            epochs: int,
            batch_size: Optional[int] = None,
            val_xs: TDataOpt = None, val_ys: TDataOpt = None,
            early_stop: Optional[EarlyStop] = None,
            plot_data_every: Optional[int] = None,
            lr: float = NotProvided(),
            shuffle: bool = False) -> Optional[List[DensityPlotData]]:
        """
        :param val_xs: validation data
        :param xs: training data.
        :param batch_size: int or None. Do not specify if any of the input data is a batched Dataset
        """
        raise NotImplementedError()

    def fit_prepare(self, ds: DS, batch_size: int, val_ds: DSOpt = None, val_contains_truth: bool = False, shuffle: bool = False) -> Tuple[DS, DSOpt, DSOpt, DSOpt, DSOpt]:
        def check_batch(unknown: Any):
            # is_dataset and is_batched and batch_size_set
            if isinstance(unknown, tf.data.Dataset) and batch_size is not None:
                # tuple
                if isinstance(unknown.element_spec, Tuple):
                    if len(unknown.element_spec[0].shape) > 1:
                        raise ValueError("batched dataset and batch_size were provided. Chose one")
                # no tuple
                elif len(unknown.element_spec.shape) > 1:
                    raise ValueError("batched dataset and batch_size were provided. Chose one")

        if isinstance(ds, DataLoader):
            ds, val_ds = ds.create_data_sets()

        check_batch(ds)
        check_batch(val_ds)

        ds_xs: DS = LearnedDistribution.Methods.cast_data(ds)
        ds_cond: DSOpt = None
        ds_val_xs: DSOpt = LearnedDistribution.Methods.cast_data(val_ds)
        ds_val_cond: DSOpt = None
        ds_val_truth_exp: DSOpt = None
        if shuffle:
            ds_xs = ds_xs.shuffle(buffer_size=len(ds_xs), reshuffle_each_iteration=True)
        if self.conditional:
            cond = lambda t: t[:self.conditional_dims]
            rest = lambda t: t[self.conditional_dims:]
            ds_cond = ds_xs.map(cond, num_parallel_calls=tf.data.AUTOTUNE)  # get cond
            ds_xs = ds_xs.map(rest, num_parallel_calls=tf.data.AUTOTUNE)  # overwrite ds_xs without cond
            if ds_val_xs is not None:
                ds_val_cond = ds_val_xs.map(cond, num_parallel_calls=tf.data.AUTOTUNE)
                if val_contains_truth:
                    ds_val_truth_exp = ds_val_xs.map(lambda t: tf.exp(t[self.conditional_dims:self.conditional_dims + 1]), num_parallel_calls=tf.data.AUTOTUNE)
                    rest = lambda t: t[self.conditional_dims + 1:]
                ds_val_xs = ds_val_xs.map(rest, num_parallel_calls=tf.data.AUTOTUNE)

        def prefetch(d: DSOpt):
            if d is None:
                return None
            return d
            return d.prefetch(tf.data.AUTOTUNE)

        ds_xs = prefetch(ds_xs)
        ds_val_xs = prefetch(ds_val_xs)
        ds_cond = prefetch(ds_cond)
        ds_val_cond = prefetch(ds_val_cond)
        ds_val_truth_exp = prefetch(ds_val_truth_exp)

        return ds_xs, ds_val_xs, ds_cond, ds_val_cond, ds_val_truth_exp

    def save(self, folder: Union[Path, str], prefix: str):
        raise NotImplementedError()

    def get_base_name_part(self, save_settings: SaveSettings) -> str:
        raise NotImplementedError()

    def get_config(self) -> LearnedConfig:
        """get a quick snapshot of the learnable parameters. EarlyStop will use these to restore the best one when training."""
        raise NotImplementedError()

    def set_training(self, training: bool):
        """In case layers have to be modified or whatever when you want to start training. set_training(False) is called after creation, loading and fit(). set_training(True)
        just before fit() """
        pass


class ConditionalLearnedDistribution(LearnedDistribution, ABC):
    def __init__(self, input_dim: int, conditional_dim: int):
        super().__init__(input_dim)
        self.conditional_dim: int = conditional_dim


class FitHistory:
    def __init__(self):
        self._d: Dict[str, List[Optional[float]]] = {}

    def add(self, key: str, value: Optional[Union[Tensor, float]]):
        if key not in self._d:
            self._d[key] = []
        if value is not None:
            self._d[key].append(float(value))
        else:
            self._d[key].append(None)

    def to_dict(self) -> Dict[str, List[float]]:
        return self._d

    def get(self, key: str) -> List[float]:
        return self._d[key]

    def truncate(self, last_epoch: int):
        self._d = {k: vs[:last_epoch] for k, vs in self._d.items()}


class EarlyStop:
    def __init__(self, monitor: str, comparison_op: Callable, patience: int, restore_best_model: bool = False):
        self.monitor: str = monitor
        self.comparison_op: Callable = comparison_op
        self.patience: int = patience
        self.best_value: Optional[float] = None
        self.last_epoch: int = 0
        self.stopped: bool = False
        self.best_learned_distribution_config: Optional[LearnedConfig] = None
        self.learned_distribution: Optional[LearnedDistribution] = None
        self.restore_best_model: bool = restore_best_model
        self.__never_stop: bool = False
        self.__has_restored: bool = False
        assert patience > 0
        # cp = tf.train.Checkpoint(optimizer=opt, model=maf)
        # cp.save()
        # co = CheckpointOptions()
        self.debug_stop_epoch: Optional[int] = None

    def after_training_ends(self, history: FitHistory):
        if self.__has_restored:
            return
        if self.restore_best_model and self.best_learned_distribution_config is not None:
            print(f"restoring model from epoch {self.last_epoch}")
            self.learned_distribution.transformed_distribution = self.best_learned_distribution_config.create().transformed_distribution
            self.__has_restored = True
            history.truncate(self.last_epoch)

    def before_training_starts(self, learned_distribution: LearnedDistribution):
        self.learned_distribution = learned_distribution

    def never_stop(self) -> EarlyStop:
        self.__never_stop = True
        return self

    def on_epoch_end(self, epoch: int, history: FitHistory) -> bool:
        """:return True when the training must be stopped"""
        if self.__never_stop:
            return False
        if self.learned_distribution is None:
            raise ReferenceError("EarlyStop.learned_distribution is None. call before_training_start().")
        if self.stopped:
            raise RuntimeError("I am EarlyStop and already stopped and yet was ignored. I am sad :'(")
        last_value = history.get(self.monitor)[-1]
        if self.best_value is None:
            self.best_value = last_value
            self.best_learned_distribution_config = self.learned_distribution.get_config()
            self.last_epoch = epoch
            return False
        if self.comparison_op(last_value, self.best_value) and (self.debug_stop_epoch is None or self.debug_stop_epoch > epoch):
            self.best_value = last_value
            self.best_learned_distribution_config = self.learned_distribution.get_config()
            self.last_epoch = epoch
            return False
        else:
            if epoch - self.last_epoch > self.patience or (self.debug_stop_epoch is not None and epoch == self.debug_stop_epoch):
                self.stopped = True
                if self.restore_best_model:
                    self.after_training_ends(history=history)
                return True
        return False

    def new(self) -> EarlyStop:
        es = EarlyStop(monitor=self.monitor, comparison_op=self.comparison_op, patience=self.patience, restore_best_model=self.restore_best_model)
        es.debug_stop_epoch = self.debug_stop_epoch
        if self.__never_stop:
            es.never_stop()
        return es


class LearnedDistributionCreator(Ser):
    def __init__(self):
        super().__init__()

    # def save(self, dist: LearnedDistribution, folder: Union[Path, str], prefix: str):
    #     t = self.get_type()
    #     if not isinstance(dist, t):
    #         raise RuntimeError(f"can only save '{t}' but '{type(dist)}' was provided.")
    #     self._save(dist=dist, folder=folder, prefix=prefix)

    def create(self, input_dim: int, conditional_dims: int, conditional_classes: List[str, int] = None) -> LearnedDistribution:
        raise NotImplementedError()

    def load(self, folder: Union[Path, str], prefix: str) -> LearnedDistribution:
        raise NotImplementedError()

    # def _save(self, dist: LearnedDistribution, folder: Union[Path, str], prefix: str):
    #     raise NotImplementedError()

    # def get_type(self) -> Type[LearnedDistribution]:
    #     raise NotImplementedError()
