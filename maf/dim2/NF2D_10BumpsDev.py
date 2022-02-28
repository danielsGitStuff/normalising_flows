from pathlib import Path

from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from common.argparser import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.DefaultDistributions import DefaultDistributions
from maf.dim2.VisualExample2D import VisualExample2D


class NF2D_10BumpsDev(VisualExample2D):
    def __init__(self):
        super().__init__("NF2D_10BumpsDev", layers=[15, 20, 25])
        self.set_minmax_square(13)
        self.print_3d_for_denses = True
        self.vmax = 'auto'

    def create_data_distribution(self) -> Distribution:
        return DefaultDistributions.create_gauss_no()

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_noise_variance=0.0, input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for
                layers in
                self.get_layers()]

    def create_data_title(self) -> str:
        return "X ~ 10xN(var; 0.3^2, 0.3^2)"


if __name__ == '__main__':
    ArgParser.parse()
    Global.set_global('results_dir', Path('results_dev'))
    NF2D_10BumpsDev().run()
