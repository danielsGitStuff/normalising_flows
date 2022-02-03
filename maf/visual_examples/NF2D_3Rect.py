from distributions.UniformMultivariate import UniformMultivariate
from pathlib import Path

from typing import List

import numpy as np
import tensorflow as tf

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.LearnedDistribution import EarlyStop
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.base import cast_to_ndarray
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.MafExperiment import MafExperiment
from maf.stuff.Foursome2DExample import Foursome2DMafExperiment


class NF2D_3Rect(Foursome2DMafExperiment):
    def __init__(self):
        super().__init__("NF2D_3Rect")

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[UniformMultivariate(input_dim=2, lows=[-2, -3], highs=[-1, -1]),
                                                                  UniformMultivariate(input_dim=2, lows=[-2, 1], highs=[-1, 3]),
                                                                  UniformMultivariate(input_dim=2, lows=[1, 1], highs=[2, 3])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [3]]

    def create_data_title(self) -> str:
        return '3 Uniforms'


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_3Rect().run()
