from typing import List

from distributions.Distribution import Distribution
from maf.VaryingParametersExperiment import Defaults
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.examples.stuff.Foursome2DExample import Foursome2DMafExperiment


class NF2D_Row3(Foursome2DMafExperiment):
    def __init__(self):
        super().__init__("NF2D_Row3")
        self.set_minmax_square(10.0)

    def create_data_distribution(self) -> Distribution:
        return Defaults.create_gauss_3_y(0.0)

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in [1, 2, 3]]

    def create_data_title(self) -> str:
        return "X ~ 3xN(0; 0.3^2, 0.3^2)"


if __name__ == '__main__':
    NF2D_Row3().run()
