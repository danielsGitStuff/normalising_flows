import numpy as np

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.base import BaseMethods
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2.VisualExample2D import VisualExample2D
from pathlib import Path
from typing import List


class NF2D_1Bumps(VisualExample2D):
    def __init__(self):
        super().__init__("NF2D_1Bumps")
        self.vmax = BaseMethods.call_func_in_process(self.data_distribution,self.data_distribution.prob,arguments=(np.array([[-2.5, -2.5]], dtype=np.float32),))[0][0]

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, -2.5], cov=[1, 1])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in
                self.get_layers()]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 2^2, 1)"


if __name__ == '__main__':
    ArgParser.parse()
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_1Bumps().run()
