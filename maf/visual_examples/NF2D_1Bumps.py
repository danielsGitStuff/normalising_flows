import numpy as np

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.visual_examples.VisualExample import VisualExample
from pathlib import Path
from typing import List


class NF2D_1Bumps(VisualExample):
    def __init__(self):
        super().__init__("NF2D_1Bumps")
        self.vmax = self.data_distribution.prob(np.array([[-2.5, -2.5]], dtype=np.float32))[0][0]

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, -2.5], cov=[1, 1])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [1, 3, 5, 10]]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 2^2, 1)"


if __name__ == '__main__':
    ArgParser.parse()
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_1Bumps().run()
