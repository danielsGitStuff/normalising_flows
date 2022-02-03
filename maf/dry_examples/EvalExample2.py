from pathlib import Path
import numpy as np
from common.globals import Global
from distributions.Distribution import Distribution
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.UniformMultivariate import UniformMultivariate
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from distributions.base import enable_memory_growth, set_gpu
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.DivergenceExperiment import DivergenceExperiment
from maf.stuff.Foursome2DExample import Foursome2DMafExperiment
from typing import List


class EvalExample2(DivergenceExperiment):

    def __init__(self):
        super().__init__('Eval2')
        self.mesh_count = 500
        self.xmin = -7
        self.xmax = 7
        self.ymin = -7
        self.ymax = 7

    def create_data_distribution(self) -> Distribution:
        d = MultimodalDistribution(input_dim=2, distributions=[
            UniformMultivariate(input_dim=2, lows=[-1, -1], highs=[1, 1]),
            UniformMultivariate(input_dim=2, lows=[-1, 2], highs=[0, 3])
        ])
        DIM: int = 2
        O = 3
        R = 3
        H_MIN = 0.6
        d = WeightedMultimodalMultivariate(input_dim=DIM)
        for i in range(7):
            weight = np.random.random() + 3
            # weight = 1.0
            src_offsets = np.random.uniform(-O, O, DIM)
            src_lows = np.random.uniform(-R, R, DIM)
            src_highs = np.random.uniform(-R, R, DIM)

            lows: List[float] = []
            highs: List[float] = []
            for o, low in zip(src_offsets, src_lows):
                # flip: bool = np.random.random() < 0.5
                # if flip:
                #     low = -low
                low += o
                high = low + np.random.uniform(H_MIN, R)
                lows.append(low)
                highs.append(high)
                # highs.append(low + 1)
                # weight = 1.0
            print(f"lows {lows}, highs {highs}, weight {weight}")
            u = UniformMultivariate(input_dim=DIM, lows=lows, highs=highs)
            d.add_d(d=u, weight=weight)
        return d

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True, batch_norm=False) for
                layers in [1, 5, 10]]

    def create_data_title(self) -> str:
        return '7 Uniforms'


if __name__ == '__main__':
    ArgParser.parse()
    set_gpu()
    enable_memory_growth()
    Global.set_seed(67)
    Global.set_global('results_dir', Path('results_artificial'))
    EvalExample2().run()
