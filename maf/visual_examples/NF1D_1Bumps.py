from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.visual_examples.VisualExample import VisualExample


class NF1D_1Bumps(VisualExample):
    def __init__(self):
        super().__init__("NF1D_1Bumps")
        self.epochs = 20

    def create_data_distribution(self) -> Distribution:
        return GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[0.5 ** 2])

    def create_data_title(self) -> str:
        return "X ~ N(-2.5; 0.5^2)"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=1, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [1, 2, 3]]


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    NF1D_1Bumps().run()
