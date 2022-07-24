import tensorflow as tf

from broken.ClassOneHot import ClassOneHot
from common.argparser import ArgParser
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.UniformMultivariate import UniformMultivariate
from distributions.conditional_categorical import ConditionalCategorical
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.dim2cond.VisualExample2DCond import VisualExample2DCond
from typing import List


class NF2DC_Test(VisualExample2DCond):
    def __init__(self):
        super().__init__(name='NF2DC_Test', layers=[2], layers_repeat=1, pool_size=1)
        self.epochs = 20

    def create_data_distribution(self) -> ConditionalCategorical:
        union = UniformMultivariate(input_dim=2, lows=[-1, -1], highs=[0, 0])
        gauss = GaussianMultivariate(input_dim=2, mus=[1, 1], cov=[1, 1])
        c = ConditionalCategorical(input_dim=2, distributions=[union, gauss])
        return c

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, conditional_dims=1, layers=2, use_tanh_made=True,
                                         class_one_hot=ClassOneHot(enabled=True, num_tokens=2, typ="int", classes=[0, 1]), cond_type='int')]


if __name__ == '__main__':
    ArgParser.parse()
    NF2DC_Test().run()
