from typing import List

import numpy as np
import tensorflow as tf

from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.LearnedDistribution import EarlyStop
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.MultimodalDistribution import MultimodalDistribution
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.examples.stuff.MafExperiment import MafExperiment
from maf.examples.stuff.Foursome2DExample import Foursome2DMafExperiment


class NF2D_2Bumps(Foursome2DMafExperiment):
    def __init__(self):
        super().__init__("NF2D_2Bumps")
        self.vmax = self.data_distribution.prob(xs=[[2.5, 2.5]])
        self.no_samples: int = 8000
        self.no_val_samples: int = 1000
        self.epochs = 50
        self.vmax = None

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, -2.5], cov=[1, 1]),
                                                                  GaussianMultivariate(input_dim=2, mus=[2.5, 2.5], cov=[1, 1])])

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 1, 1) & N(2.5, 2.5; 1, 1)"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in [3]]




if __name__ == '__main__':
    NF2D_2Bumps().run()
