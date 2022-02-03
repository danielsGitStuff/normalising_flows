from typing import List

import numpy as np
import tensorflow as tf

from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.LearnedDistribution import EarlyStop
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.UniformMultivariate import UniformMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.MafExperiment import MafExperiment
from maf.stuff.Foursome2DExample import Foursome2DMafExperiment


class U_NF2D_2Bumps(Foursome2DMafExperiment):
    def __init__(self):
        super().__init__("U_NF2D_2Bumps")
        self.batch_size = 512
        self.vmax = self.data_distribution.prob(xs=[[2.5, 2.5]])
        self.no_samples: int = 80000
        self.no_val_samples: int = 3000
        self.epochs = 200
        self.vmax = None
        self.divergence_step_size = 0.1
        self.divergence_half_width = 3.5

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2,
                                      distributions=[UniformMultivariate(input_dim=2, lows=[-3, -1], highs=[-1, 1]),
                                                     UniformMultivariate(input_dim=2, lows=[1, -1], highs=[3, 1])])

    def create_data_title(self) -> str:
        return "X ~ U(-2.5, -2.5; -1, -1) & N(1, 1; 2.5, 2.5)"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(use_tanh_made=True, input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in [2, 5, 8]]


if __name__ == '__main__':
    U_NF2D_2Bumps().run()