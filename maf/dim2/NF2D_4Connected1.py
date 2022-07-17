from pathlib import Path

from typing import List

from common.globals import Global
from distributions.distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from common.argparser import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.DefaultDistributions import DefaultDistributions
from maf.dim2.VisualExample2D import VisualExample2D


class NF2D_4Connected1(VisualExample2D):
    def __init__(self):
        super().__init__("NF2D_4Connected1")
        self.set_minmax_square(10.0)

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-5, 1.5], cov=[1, 4]),
                                                                  GaussianMultivariate(input_dim=2, mus=[-1.5, 1.5], cov=[3, 1]),
                                                                  GaussianMultivariate(input_dim=2, mus=[1.5, -1.5], cov=[1, 3]),
                                                                  GaussianMultivariate(input_dim=2, mus=[3, -5], cov=[4, 1])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in
                self.get_layers()]

    def create_data_title(self) -> str:
        return "X ~ 4xN(var; var)"


if __name__ == '__main__':
    ArgParser.parse()
    NF2D_4Connected1().run()
