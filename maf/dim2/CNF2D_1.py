from broken.ClassOneHot import ClassOneHot
from distributions.ConditionalCategoricalOld import ConditionalCategoricalOld
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate
from pathlib import Path

from typing import List

from common.globals import Global
from distributions.distribution import Distribution
from distributions.UniformMultivariate import UniformMultivariate
from common.argparser import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2.VisualExample2D import VisualExample2D


class CNF2D_1(VisualExample2D):
    def __init__(self):
        super().__init__("CNF2D_1", layers=[2, 3], layers_repeat=1, pool_size=1)

    def create_data_distribution(self) -> Distribution:
        c = ConditionalCategoricalOld(input_dim=2, conditional_dims=1)
        c.add_d(d=UniformMultivariate(input_dim=2, lows=[-1, -2], highs=[1, 2]), weight=1)
        c.add_d(d=WeightedMultimodalMultivariate(input_dim=2)
                .add_d(UniformMultivariate(input_dim=2, lows=[-3, -3], highs=[-2, 3]), weight=1 / 4)
                .add_d(UniformMultivariate(input_dim=2, lows=[2, -3], highs=[3, 3]), weight=1 / 4)
                .add_d(UniformMultivariate(input_dim=2, lows=[-2, -3], highs=[2, -2]), weight=1), weight=1)
        return c

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True, conditional_dims=1,
                                         class_one_hot=ClassOneHot(enabled=True, num_tokens=2, typ="int", classes=[1, 2])) for layers in self.get_layers()]
        # return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in
        #         self.get_layers()]

    def create_data_title(self) -> str:
        return "Testi"

    def results_dir_name(self) -> str:
        return "results_artificial_dim2_cond"


if __name__ == '__main__':
    ArgParser.parse()
    CNF2D_1().run()
