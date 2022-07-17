from typing import List

from distributions.distribution import Distribution
from distributions.UniformMultivariate import UniformMultivariate
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from common.argparser import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2.VisualExample2D import VisualExample2D


class NF2D_1L(VisualExample2D):
    def __init__(self):
        super().__init__("NF2D_1L", layers=[10, 20], layers_repeat=1)
        # self.epochs = 1
        self.vmax = 'auto'
        # self.vmax = 0.02

    def create_data_distribution(self) -> Distribution:
        return WeightedMultimodalMultivariate(input_dim=2) \
            .add_d(UniformMultivariate(input_dim=2, lows=[-3, -3], highs=[-2, 3]), weight=1 / 4) \
            .add_d(UniformMultivariate(input_dim=2, lows=[-2, -3], highs=[2, -2]), weight=1 / 6)

    def create_data_title(self) -> str:
        return "X ~ L"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in
                self.get_layers()]


if __name__ == '__main__':
    ArgParser.parse()
    NF2D_1L().run()
