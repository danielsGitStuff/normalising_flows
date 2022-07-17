from pathlib import Path

from typing import List

from common.globals import Global
from distributions.distribution import Distribution
from distributions.UniformMultivariate import UniformMultivariate
from common.argparser import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2.VisualExample2D import VisualExample2D


class NF2D_1RectL2(VisualExample2D):
    def __init__(self):
        super().__init__("NF2D_1RectL2", layers=[2], layers_repeat=2)

    def create_data_distribution(self) -> Distribution:
        return UniformMultivariate(input_dim=2, lows=[-1, -2], highs=[1, 2])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in self.get_layers()]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 2^2, 1)"


if __name__ == '__main__':
    ArgParser.parse()
    NF2D_1RectL2().run()
