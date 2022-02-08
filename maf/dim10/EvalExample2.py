from pathlib import Path

import numpy as np
from typing import List

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.UniformMultivariate import UniformMultivariate
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim10.VisualRandomExample import VisualRandomExample


class EvalExample2(VisualRandomExample):

    def __init__(self):
        super().__init__('EvalExample2')
        self.mesh_count = 500
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10

    def create_data_distribution(self) -> Distribution:
        DIM: int = 2
        O = 3
        R = 3
        H_MIN = 0.6
        rng = np.random.default_rng()
        d = WeightedMultimodalMultivariate(input_dim=DIM)
        for i in range(7):
            weight = rng.random() + 3
            src_offsets = rng.uniform(-O, O, DIM)
            src_lows = rng.uniform(-R, R, DIM)

            lows: List[float] = []
            highs: List[float] = []
            for o, low in zip(src_offsets, src_lows):
                low += o
                high = low + rng.uniform(H_MIN, R)
                lows.append(low)
                highs.append(high)
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
    Global.set_seed(50)
    EvalExample2().run()
