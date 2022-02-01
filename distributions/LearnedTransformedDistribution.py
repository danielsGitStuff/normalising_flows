from __future__ import annotations

import pandas as pd
from typing import Union, Optional

import tensorflow as tf
import os
from pathlib import Path

from common import jsonloader
from distributions.LearnedDistribution import LearnedDistribution
from tensorflow_probability.python import distributions as tfd

from distributions.base import TTensor, TTensorOpt


class LearnedTransformedDistribution(LearnedDistribution):
    @staticmethod
    def can_load_from(folder: Union[Path, str], prefix: str) -> bool:
        complete_prefix = str(Path(folder, prefix))
        json_file = f"{complete_prefix}.model.json"
        checkpoint_file = f"{complete_prefix}.data-00000-of-00001"
        return os.path.exists(json_file) and os.path.exists(checkpoint_file)

    @staticmethod
    def load(folder: Union[Path, str], prefix: str) -> LearnedTransformedDistribution:
        complete_prefix = str(Path(folder, prefix))
        json_file = f"{complete_prefix}.model.json"
        print(f"loading learned distribution from '{json_file}'")
        dis: LearnedTransformedDistribution = jsonloader.load_json(json_file)
        checkpoint = tf.train.Checkpoint(model=dis.transformed_distribution)
        checkpoint.restore(complete_prefix)
        dis.set_training(False)
        return dis

    def build_transformation(self):
        raise NotImplementedError()

    def after_deserialize(self):
        self.build_transformation()

    def __init__(self, input_dim: int, conditional_dims: int):
        super().__init__(input_dim, conditional_dims=conditional_dims)
        self.transformed_distribution: Optional[tfd.TransformedDistribution] = None
        self.ignored.add("transformed_distribution")

    def save(self, folder: Union[Path, str], prefix: str):
        os.makedirs(folder, exist_ok=True)
        complete_prefix = str(Path(folder, prefix))
        json_file = f"{complete_prefix}.model.json"
        # tf checkpoint
        check = tf.train.Checkpoint(model=self.transformed_distribution)
        check.write(file_prefix=complete_prefix)
        # history
        if self.history is not None:
            history: pd.DataFrame = pd.DataFrame(self.history.to_dict())
            history.to_csv(f"{complete_prefix}.history.csv", index=False)
        # config
        self.to_json(json_file, pretty_print=True)

    def calculate_us(self, xs: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        xs, cond = self.extract_xs_cond(xs, cond)
        return self.calculate_us(xs, cond)

    def calculate_xs(self, us: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        xs, cond = self.extract_xs_cond(us, cond)
        return self.calculate_xs(us, cond)

    def calculate_det_density(self, xs: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        xs, cond = self.extract_xs_cond(xs, cond)
        return self._calculate_det_density(xs, cond)

    def _calculate_us(self, xs: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        raise NotImplementedError()

    def _calculate_xs(self, us: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        raise NotImplementedError()

    def _calculate_det_density(self, xs: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        raise NotImplementedError()
