from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim1.VisualExample1D import VisualExample1D
from typing import List


class NF1D_2Bumps2(VisualExample1D):
    def __init__(self):
        super().__init__("NF1D_2Bumps2", layers_repeat=1, layers=[1, 3])
        self.vmax = .6
        self.epochs = 10

    def create_data_distribution(self) -> Distribution:
        return WeightedMultimodalMultivariate(input_dim=1) \
            .add_d(GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[0.5 ** 2]), weight=1) \
            .add_d(GaussianMultivariate(input_dim=1, mus=[2.5], cov=[.5 ** 2]), weight=2)

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=1, layers=layers, activation="relu", hidden_shape=[10, 10], norm_layer=True, use_tanh_made=True) for layers in self.get_layers()]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5; 0.5^2)/3 & N(2.5; 0.5^2) * 2/3"


if __name__ == '__main__':
    ArgParser.parse()
    NF1D_2Bumps2().run()
