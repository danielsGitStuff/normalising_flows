from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.UniformMultivariate import UniformMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.visual_examples.VisualExample import VisualExample


class NF2D_1Rect(VisualExample):
    def __init__(self):
        super().__init__("NF2D_1Rect")

    def create_data_distribution(self) -> Distribution:
        return UniformMultivariate(input_dim=2, lows=[-1, -2], highs=[1, 2])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [1, 2, 3]]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 2^2, 1)"


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_1Rect().run()
