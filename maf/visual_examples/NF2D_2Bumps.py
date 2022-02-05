from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.visual_examples.VisualExample import VisualExample


class NF2D_2Bumps(VisualExample):
    def __init__(self):
        super().__init__("NF2D_2Bumps")
        self.vmax = self.data_distribution.prob(xs=[[2.5, 2.5]])

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, -2.5], cov=[1, 1]),
                                                                  GaussianMultivariate(input_dim=2, mus=[2.5, 2.5], cov=[1, 1])])

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 1, 1) & N(2.5, 2.5; 1, 1)"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [1, 3, 5, 10]]


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_2Bumps().run()
