from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2.VisualExample2D import VisualExample2D


class NF1D_2Bumps(VisualExample2D):
    def __init__(self):
        super().__init__("NF1D_2Bumps")
        self.vmax = 0.5
        self.print_3d_for_denses = False
        self.epochs = 30

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=1, distributions=[GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[0.5 ** 2]),
                                                                  GaussianMultivariate(input_dim=1, mus=[2.5], cov=[0.5 ** 2])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=1, layers=layers, activation="relu", hidden_shape=[10, 10], norm_layer=True, use_tanh_made=True) for layers in self.get_layers()]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5; 0.5^2) & N(2.5; 0.5^2)"


if __name__ == '__main__':
    ArgParser.parse()
    Global.set_global('results_dir', Path('results_artificial'))
    NF1D_2Bumps().run()