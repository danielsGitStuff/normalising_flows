from __future__ import annotations

from numbers import Number

from tensorflow import Tensor

from typing import Optional, List, Callable, Union

import numpy as np
import tensorflow as tf
from matplotlib.colors import Colormap, TwoSlopeNorm

from common import util
from common.NotProvided import NotProvided
from common.jsonloader import SerSettings
from distributions.distribution import Distribution, TfpD, HeatmapCreator
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.base import TTensor, TTensorOpt
from distributions.density_plot_data import DensityPlotData


class ConditionalCategorical(Distribution):
    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        return self._call_ordered(f_name="log_likelihoods", xs=xs, cond=cond)

    def __init__(self, input_dim: int = NotProvided(), distributions: List[Distribution] = NotProvided(), d_probs: List[float] = NotProvided()):
        super(ConditionalCategorical, self).__init__(input_dim=input_dim, conditional_dims=1)
        self.conditional_sample = True
        self.distributions: List[Distribution] = distributions
        self.d_probs: List[float] = d_probs
        self._init()

    @property
    def heatmap_creator(self) -> HeatmapCreator:
        hm = ConditionalHeatmapCreator(self, cond_values=list(range(len(self.distributions))))
        return hm

    def _create_base_distribution(self) -> Optional[TfpD]:
        pass

    def sample(self, size: int = 1, cond: TTensorOpt = None, batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        if cond is None:
            raise RuntimeError("No conditional provided when called sample() on a conditional distribution")
        if self.tfd_distribution is None:
            self.create_base_distribution()
        cond = self.cast_2_input(cond, event_dim=self.conditional_dims)
        # sort first
        swap_indices = tf.argsort(cond[:, 0])
        reverse_indices = tf.argsort(swap_indices)
        cond_sorted = tf.gather(cond, swap_indices)
        result: np.ndarray = self.batch_call_sample(size=size, cond=cond_sorted, batch_size=batch_size, **kwargs)
        result: Tensor = tf.gather(result, reverse_indices)
        return self.cast_2_float_ndarray(result)

    def _sample(self, size: int = 1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        result: List[np.ndarray] = []
        unique_indices, idx, counts = tf.unique_with_counts(cond[:, 0])
        for dist_index, count in zip(unique_indices, counts):
            dist_index = int(dist_index)
            d: Distribution = self.distributions[dist_index]
            samples = d.sample(count)
            result.append(samples)
        result: np.ndarray = np.concatenate(result)
        return self.cast_2_float_ndarray(result)

    def _call_ordered(self, f_name: str, xs: TTensor, cond: TTensorOpt) -> np.ndarray:
        """xs and cond are already batched!"""
        result: List[np.ndarray] = []
        unique_indices, idx, counts = tf.unique_with_counts(cond[:, 0])
        tensor_index = 0
        for dist_index, count in zip(unique_indices, counts):
            dist_index = int(dist_index)
            t = xs[tensor_index:tensor_index + count]
            f: Callable[[TTensor, TTensorOpt, Optional[int]], np.ndarray] = getattr(self.distributions[dist_index], f_name)
            result.append(f(t, None, None))
            tensor_index += count
        result: np.ndarray = np.concatenate(result)
        return self.cast_2_likelihood(xs, result)

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        return self._call_ordered(f_name="likelihoods", xs=xs, cond=cond)

    def likelihoods(self, xs: TTensor, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        if self.tfd_distribution is None:
            self.create_base_distribution()
        xs, cond = self.extract_xs_cond(xs, cond)
        xs = self.cast_2_input(xs, event_dim=self.input_dim)
        cond = self.cast_2_input(cond, event_dim=self.conditional_dims)
        """sort the list regarding the conditional. so all inputs belonging to one distribution are adjacent"""
        swap_indices = tf.argsort(cond[:, 0])
        reverse_indices = tf.argsort(swap_indices)
        xs_sorted = tf.gather(xs, swap_indices)
        cond_sorted = tf.gather(cond, swap_indices)
        """call the actual calculation of the likelihoods"""
        result = self.batch_call(self._likelihoods, xs_sorted, cond_sorted, batch_size)
        """undo the sorting"""
        result = tf.gather(result, reverse_indices)
        return self.cast_2_likelihood(xs, result)

    def _init(self) -> bool:
        """check if everything is provided"""
        if NotProvided.is_provided(self.distributions):
            if NotProvided.is_not_provided(self.d_probs):
                equal_prob: float = 1.0 / len(self.distributions)
                self.d_probs = [equal_prob] * len(self.distributions)
        if NotProvided.is_not_provided(self.distributions) or NotProvided.is_not_provided(self.d_probs):
            return False
        return True


# if __name__ == '__main__':
#     SerSettings.enable_testing_mode()
#     util.set_seed(4)
#     c = ConditionalCategorical(input_dim=1, distributions=[GaussianMultivariate(1, mus=[-2], cov=[1]), GaussianMultivariate(1, mus=[2], cov=[1])])
#     samples = c.sample(10, batch_size=4)
#     print("SAMPLES")
#     print(samples)
#     print("")
#     ls = c.likelihoods(samples, batch_size=4)
#     print("LIKELIHOODS")
#     print(ls)


class ConditionalDensityPlotData(DensityPlotData):
    def __init__(self):
        super().__init__(np.empty((1, 1), dtype=np.float32), 0)
        self.conditionals: List[DensityPlotData] = []

    def add_conditional(self, density_plot_data: DensityPlotData):
        self.conditionals.append(density_plot_data)

    def print_yourself(self, ax, vmax: Optional[float] = None, vmin: Optional[float] = None, cmap: Optional[Colormap] = None, legend: bool = NotProvided,
                       norm: Optional[TwoSlopeNorm] = None):
        print("asdrrrr")


class ConditionalHeatmapCreator(HeatmapCreator):
    def __init__(self, dist: ConditionalCategorical = NotProvided(), cond_values: List[Union[float, int]] = NotProvided()):
        super().__init__(dist)
        self.dist: ConditionalCategorical = dist
        self.cond_values: List[Number] = cond_values

    def heatmap_2d_data(self, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=200, suptitle: str = None, title: str = None,
                        columns: Optional[List[str]] = NotProvided(), true_distribution: Optional[Distribution] = None) -> DensityPlotData:
        """plot_data.append(heatmap_2d_data(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, suptitle="p(x) sampled from Z", title=f"after {i} epochs"))"""
        x = tf.linspace(xmin, xmax, mesh_count)
        y = tf.linspace(ymin, ymax, mesh_count)
        X, Y = tf.meshgrid(x, y)
        result = ConditionalDensityPlotData()
        for cond in self.cond_values:
            cond: Union[float, int] = cond
            concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
            concatenated_mesh_coordinates = tf.concat([concatenated_mesh_coordinates, tf.fill((concatenated_mesh_coordinates.shape[0], self.dist.conditional_dims), float(cond))],
                                                      1)
            prob = self.dist.prob(concatenated_mesh_coordinates, batch_size=10000)
            probs = tf.reshape(prob, (mesh_count, mesh_count))
            probs = probs.numpy()
            # probs = np.flip(probs,axis=(1,0))
            probs = np.rot90(probs)

            truth: Optional[np.ndarray] = None
            if true_distribution is not None:
                truth = true_distribution.prob(concatenated_mesh_coordinates, batch_size=10000)
                truth = truth.reshape((mesh_count, mesh_count))
                truth = np.rot90(truth)

            data = DensityPlotData(values=probs, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, type="hm", mesh_count=mesh_count, suptitle=suptitle, title=title, columns=columns,
                                   truth=truth)
            result.add_conditional(data)
        return result


if __name__ == '__main__':
    SerSettings.enable_testing_mode()
    util.set_seed(4)
    c = ConditionalCategorical(2, distributions=[GaussianMultivariate(2, mus=[0, 0], cov=[1, 1]), GaussianMultivariate(2, mus=[20, 20], cov=[1, 1])])
    samples = c.sample(7, cond=[1, 0, 0, 0, 1])
    print(samples)
