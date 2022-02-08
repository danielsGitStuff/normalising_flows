from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2.VisualExample2D import VisualExample2D


class NF1D_1Bumps(VisualExample2D):
    def __init__(self):
        super().__init__("NF1D_1Bumps", layers=[1, 3])
        self.epochs = 20

    def create_data_distribution(self) -> Distribution:
        return GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[0.5 ** 2])

    def create_data_title(self) -> str:
        return "X ~ N(-2.5; 0.5^2)"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=1, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in
                self.get_layers()]


if __name__ == '__main__':
    ArgParser.parse()
    Global.set_global('results_dir', Path('results_artificial'))
    NF1D_1Bumps().run()
