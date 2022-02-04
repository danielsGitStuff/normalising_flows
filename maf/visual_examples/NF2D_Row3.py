from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.VaryingParametersExperiment import Defaults
from maf.visual_examples.VisualExample import VisualExample


class NF2D_Row3(VisualExample):
    def __init__(self):
        super().__init__("NF2D_Row3")
        self.set_minmax_square(10.0)

    def create_data_distribution(self) -> Distribution:
        return Defaults.create_gauss_3_y(0.0)

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [1, 2, 3]]

    def create_data_title(self) -> str:
        return "X ~ 3xN(0; 0.3^2, 0.3^2)"


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_Row3().run()
