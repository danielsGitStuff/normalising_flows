from pathlib import Path

from typing import List

import numpy as np
import tensorflow as tf

from common.globals import Global
from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.LearnedDistribution import EarlyStop
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.base import cast_to_ndarray
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.MafExperiment import MafExperiment
from maf.visual_examples.VisualExample import VisualExample


class NF2D_1Bumps(VisualExample):
    def __init__(self):
        super().__init__("NF2D_1Bumps")
        self.vmax = self.data_distribution.prob(np.array([[-2.5, -2.5]], dtype=np.float32))[0][0]

    def create_data_distribution(self) -> Distribution:
        return MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, -2.5], cov=[1, 1])])

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True, use_tanh_made=True) for layers in [1, 3, 5, 10]]

    def create_data_title(self) -> str:
        return "X ~ N(-2.5, -2.5; 2^2, 1)"


class NF2D_1BumpsOlde(MafExperiment):
    def __init__(self):
        super().__init__("NF2D_1Bumps")

    def _run(self):
        mus = [-2.5, -2.5]
        data_distribution: Distribution = MultimodalDistribution(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=mus, cov=[2 ** 2, 1])])
        vmax = GaussianMultivariate(input_dim=2, mus=[0.0, 0.0], cov=[1.0, 1.0]).prob(np.array([[0.0, 0.0]], dtype=np.float32))[0][0]

        xs: np.ndarray = data_distribution.sample(8000)
        val_xs: np.ndarray = data_distribution.sample(1000)

        test_samples_us: np.ndarray = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        self.hm(dist=data_distribution, title="X ~ N(-2.5, -2.5; 2^2, 1)", vmax=vmax)
        for layers, (row, column) in zip(range(1, 4), [(0, 1), (1, 0), (1, 1)]):
            es = EarlyStop(monitor="val_loss", comparison_op=tf.less, patience=10, restore_best_model=True)
            prefix = self.maf_prefix(layers)
            if LearnedTransformedDistribution.can_load_from(self.cache_dir, prefix=prefix):
                maf: MaskedAutoregressiveFlow = LearnedTransformedDistribution.load(self.cache_dir, prefix=prefix)
            else:
                maf = MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True)
                maf.fit(xs=xs, batch_size=128, epochs=200, val_xs=val_xs, early_stop=es)
                maf.save(self.cache_dir, prefix=prefix)
            title = f"MAF {layers}L"
            self.hm(dist=maf, title=title, vmax=vmax)
            self.cut(maf=maf, pre_title=title)
            test_samples_xs = cast_to_ndarray(maf.tfd_distribution.bijector.forward(test_samples_us))
            print(f"u=(0,0) -> ({test_samples_xs[0][0]}, {test_samples_xs[0][1]})")


if __name__ == '__main__':
    ArgParser.parse()
    Global.set_global('results_dir', Path('results_artificial'))
    NF2D_1Bumps().run()

# code may fail learning
# for run in range(10):
#     xs: np.ndarray = source_distribution.sample(8000)
#     val_xs: np.ndarray = source_distribution.sample(1000)
#
#     fig, axs = StaticMethods.default_fig(2, 2)
#     source_distribution.heatmap_creator.heatmap_2d_data(title="source").print_yourself(axs[0][0])
#     layers = 3
#     es = EarlyStop(monitor="val_loss", comparison_op=tf.less, patience=10, restore_best_model=True)
#     prefix = f"learn2bumps.maf{layers}"
#     prefix = f"{prefix}_{run}"
#     if LearnedTransformedDistribution.can_load_from(cache_dir, prefix=prefix):
#         maf: MaskedAutoregressiveFlow = LearnedTransformedDistribution.load(cache_dir, prefix=prefix)
#     else:
#         maf = MaskedAutoregressiveFlow(input_dim=2, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=False)
#         maf.fit(xs=xs, batch_size=128, epochs=10000, val_xs=val_xs, early_stop=es)
#         maf.save(cache_dir, prefix=prefix)
#     maf.heatmap_creator.heatmap_2d_data(title=f"learned {layers}L").print_yourself(axs[1][1])
#
#     plt.tight_layout()
#     plt.savefig(f"Learn2Bumps{run}.png", transparent=True)
