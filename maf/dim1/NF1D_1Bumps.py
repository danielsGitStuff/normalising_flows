from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from common.argparser import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim1.VisualExample1D import VisualExample1D
from typing import List


class NF1D_1Bumps(VisualExample1D):
    def __init__(self):
        super().__init__("NF1D_1Bumps", layers=[1, 3])

    def create_data_distribution(self) -> Distribution:
        return GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[0.5 ** 2])

    def create_data_title(self) -> str:
        return "X ~ N(-2.5; 0.5^2)"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=1, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in
                self.get_layers()]


if __name__ == '__main__':
    ArgParser.parse()
    NF1D_1Bumps().run()
