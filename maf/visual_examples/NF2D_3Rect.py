from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.UniformMultivariate import UniformMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.visual_examples.VisualExample import VisualExample


class NF2D_3Rect(VisualExample):
    def __init__(self):
        super().__init__("NF2D_3Rect")

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[UniformMultivariate(input_dim=2, lows=[-2, -3], highs=[-1, -1]),
                                                                  UniformMultivariate(input_dim=2, lows=[-2, 1], highs=[-1, 3]),
                                                                  UniformMultivariate(input_dim=2, lows=[1, 1], highs=[2, 3])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [3]]

    def create_data_title(self) -> str:
        return '3 Uniforms'


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_3Rect().run()
