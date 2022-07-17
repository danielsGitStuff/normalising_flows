from pathlib import Path

from typing import List

from common.globals import Global
from distributions.distribution import Distribution
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.UniformMultivariate import UniformMultivariate
from common.argparser import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2.VisualExample2D import VisualExample2D


class NF2D_4Rect(VisualExample2D):
    def __init__(self):
        super().__init__("NF2D_4Rect")

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[UniformMultivariate(input_dim=2, lows=[-2, -3], highs=[-1, -1]),
                                                                  UniformMultivariate(input_dim=2, lows=[-2, 1], highs=[-1, 3]),
                                                                  UniformMultivariate(input_dim=2, lows=[1, 1], highs=[2, 3]),
                                                                  UniformMultivariate(input_dim=2, lows=[1, -3], highs=[2, -1])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in self.get_layers()]

    def create_data_title(self) -> str:
        return ''


if __name__ == '__main__':
    ArgParser.parse()
    NF2D_4Rect().run()
