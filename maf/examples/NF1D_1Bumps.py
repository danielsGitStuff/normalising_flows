from typing import List

from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.examples.stuff.Foursome2DExample import Foursome2DMafExperiment


class NF1D_1Bumps(Foursome2DMafExperiment):
    def __init__(self):
        super().__init__("NF1D_1Bumps")

    def create_data_distribution(self) -> Distribution:
        return GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[0.5 ** 2])

    def create_data_title(self) -> str:
        return "X ~ N(-2.5; 0.5)"

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        return [MaskedAutoregressiveFlow(input_dim=1, layers=layers, activation="relu", hidden_shape=[200, 200], norm_layer=True) for layers in [1, 2, 3]]

    # def _run(self):
    #     data_distribution: Distribution = MultimodalDistribution(input_dim=1, distributions=[GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[0.5 ** 2])])
    #     xs: np.ndarray = data_distribution.sample(8000)
    #     val_xs: np.ndarray = data_distribution.sample(1000)
    #
    #     test_samples_us: np.ndarray = np.array([[0], [1]], dtype=np.float32)
    #
    #     self.hm(dist=data_distribution, title="X ~ N(-2.5; 0.5^2)", vmax=1.1)
    #     for layers, (row, column) in zip(range(1, 4), [(0, 1), (1, 0), (1, 1)]):
    #         es = EarlyStop(monitor="val_loss", comparison_op=tf.less, patience=10, restore_best_model=True)
    #         prefix = self.maf_prefix(layers)
    #         if LearnedTransformedDistribution.can_load_from(self.cache_dir, prefix=prefix):
    #             maf: MaskedAutoregressiveFlow = LearnedTransformedDistribution.load(self.cache_dir, prefix=prefix)
    #         else:
    #             maf = MaskedAutoregressiveFlow(input_dim=1, layers=layers, activation="relu", hidden_shape=[10, 10], norm_layer=True)
    #             maf.fit(dataset=xs, batch_size=128, epochs=200, val_xs=val_xs, early_stop=es)
    #             maf.save(self.cache_dir, prefix=prefix)
    #         self.hm(dist=maf, title=f"MAF {layers}L", vmax=1.1)
    #         test_samples_xs = cast_to_ndarray(maf.transformed_distribution.bijector.forward(test_samples_us))
    #         print(f"u=(0) -> {test_samples_xs[0][0]}")


if __name__ == '__main__':
    NF1D_1Bumps().run()
