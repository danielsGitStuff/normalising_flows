from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.UniformMultivariate import UniformMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.Foursome2DExample import Foursome2DMafExperiment
from pathlib import Path
from typing import List


class NF2D_1Rect(Foursome2DMafExperiment):
    def __init__(self):
        super().__init__("NF2D_1Rect")
        print(self.vmax)
        self.no_samples: int = 8000
        self.no_val_samples: int = 1000
        self.epochs = 50
        self.vmax = 'auto'

    def create_data_distribution(self) -> Distribution:
        # return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, -2.5], cov=[2 ** 2, 1])])
        return UniformMultivariate(input_dim=2, lows=[-1, -2], highs=[1, 2])
        return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, -2.5], cov=[1, 1])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in [1, 2, 3]]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 2^2, 1)"


if __name__ == '__main__':
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_1Rect().run()
