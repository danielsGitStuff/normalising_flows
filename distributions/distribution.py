from __future__ import annotations

import math
import setproctitle

from common.globals import Global
from typing import List, Union, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.stats import multivariate_normal
from tensorflow import Tensor
from tensorflow_probability.python.distributions import Distribution as TD
import tensorflow_probability.python.distributions as tfpds
from common import jsonloader
from common.NotProvided import NotProvided
from common.jsonloader import Ser
from common.util import Runtime
from distributions.base import TTensor, TTensorOpt, cast_to_tensor, cast_to_ndarray, BaseMethods
from distributions.density_plot_data import DensityPlotData
from maf.DS import DS

TfpD = tfpds.Distribution


class Distribution(Ser):
    def __init__(self, input_dim: int = NotProvided(), conditional_dims: int = 0):
        super().__init__()
        self.input_dim: int = input_dim
        self.conditional_dims: int = conditional_dims
        self.conditional: bool = self.conditional_dims > 0
        """conditionals can be sampled too, so they do not have to be provided when calling sample()"""
        self.conditional_sample: bool = False
        self.tfd_distribution: Optional[TD] = None
        """deprecated"""
        self.conditional_producer_function: Optional[Callable[[Distribution, List[float]], Distribution]] = None
        self.ignored.add('tfd_distribution')

    def kl_divergence(self: Distribution, other: Distribution, no_of_samples: int = 100000, batch_size: int = 100000) -> float:
        rest = no_of_samples
        kl: float = 0.0
        while rest > 0:
            take = min(rest, batch_size)
            samples_self = self.sample(take)
            log_ps_self = self.log_prob(samples_self)
            log_ps_other = other.log_prob(samples_self)
            kl_sum = np.sum(log_ps_self * (log_ps_self - log_ps_other))
            kl += kl_sum
            rest -= take
        return kl

    def make_conditional(self, conditional_dims: int, producer_function: Optional[Callable[[Distribution, List[float]], Distribution]]):
        self.conditional_producer_function = producer_function
        self.conditional_dims = conditional_dims
        self.conditional = self.conditional_dims > 0
        return self

    def create_base_distribution(self):
        if self.tfd_distribution is None:
            self.tfd_distribution = self._create_base_distribution()

    def _create_base_distribution(self) -> Optional[TfpD]:
        raise NotImplementedError()

    def check_condition(self, cond: TTensorOpt):
        if self.conditional and cond is None and not self.conditional_sample:
            raise ValueError("Distribution is conditional but no condition was provided")
        if not self.conditional and cond is not None:
            raise ValueError("Distribution is NOT conditional but condition was provided")

    def extract_xs_cond(self, xs: Union[TTensor, Tuple[Tensor, Tensor]], cond: TTensorOpt = None) -> Tuple[Tensor, Optional[Tensor]]:
        """convenience method to unpack xs and cond to (xs, conditional) depending on whether this Distribution is conditional"""
        cond_provided = BaseMethods.is_conditional_data(xs, dim=self.input_dim, cond_dim=self.conditional_dims) or cond is not None  # isinstance(xs, Tuple) or cond is not None
        if self.conditional and not cond_provided:
            raise ValueError("Distribution is conditional but no condition was provided")
        if not self.conditional and cond_provided:
            raise ValueError("Distribution is NOT conditional but condition was provided")
        # extract cond
        if isinstance(xs, Tuple) and cond is not None:
            raise ValueError("Conditional was provided via 'cond' and 'xs'")
        if cond is not None:
            x, _ = cast_to_tensor(xs)
            c, _ = cast_to_tensor(cond)
            return x, c
        elif cond is None and self.conditional and len(xs.shape) == 2 and xs.shape[1] == self.input_dim + self.conditional_dims:
            x, _ = cast_to_tensor(xs[:, :-1])
            c, _ = cast_to_tensor(xs[:, -1:])
            return x, c
        else:
            return cast_to_tensor(xs)
        return BaseMethods.extract_xs_cond(xs, cond)

    def cast_2_likelihood(self, input_tensor: TTensor, result: TTensor) -> np.ndarray:
        """correct shape too"""
        result = cast_to_ndarray(result)
        return np.array(result, dtype=np.float32).reshape(input_tensor.shape[0], 1)

    def cast_2_float_ndarray(self, ls: List[Union[float, int]]) -> np.ndarray:
        if isinstance(ls, NotProvided):
            return ls
        return np.array(ls, dtype=np.float32)

    def cast_2_input(self, values: [TTensor, List[float], List[List[float]]], event_dim: int, dtype=tf.float32) -> Optional[Tensor]:
        if event_dim == 0 or values is None:
            return None
        xs = tf.constant(values, dtype=tf.float32)
        if len(xs.shape) == 1:
            xs = tf.expand_dims(xs, 1)
        if not xs.shape[1] == event_dim:
            raise ValueError(f"xs got wrong shape: {xs.shape}, expected (var,{event_dim})")
        return xs

    def batch_call(self, call_fn: Callable[[TTensor, TTensorOpt], np.ndarray], xs: TTensorOpt, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        """this may prevent crashing because of insufficient memory. If batch_size is None, call_fn will return immediately, otherwise xs and cond will be batched and fed
        to call_fn."""
        if self.tfd_distribution is None:
            self.create_base_distribution()
        if batch_size is None:
            return call_fn(xs, cond)
        results = []
        if cond is None:
            d: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(xs)
        else:
            # d: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([xs, cond])
            d: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(tf.concat([xs, cond], axis=1))
        d = d.batch(batch_size=batch_size)
        for batch in d:
            x, c = self.extract_xs_cond(batch, None)
            results.append(call_fn(x, c))
        r = np.concatenate(results, axis=0)
        return r

    def batch_call_sample(self, size: int, cond: TTensorOpt = None, batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        """this may prevent crashing because of insufficient memory. If batch_size is None, call_fn will return immediately, otherwise xs and cond will be batched and fed
        to call_fn."""
        if self.tfd_distribution is None:
            self.create_base_distribution()
        self.check_condition(cond=cond)
        if batch_size is None:
            samples = self._sample(size=size, cond=cond)
            return self.cast_2_float_ndarray(samples)
        results = []
        if cond is None:
            rest = size
            batch_count = 0
            while rest > 0:
                s = min(batch_size, rest)
                r = Runtime(f"sampling batch {batch_count + 1}/{math.ceil(size / batch_size)} with {s} entries").start()
                results.append(self._sample(size=s))
                r.stop().print()
                batch_count += 1
                rest = rest - s
        else:
            if not isinstance(cond, tf.data.Dataset):
                cond, _ = cast_to_tensor(cond)
                cond = DS.from_tensor_slices(cond)
            conds = cond.batch(batch_size=batch_size)
            for i, batched_cond in enumerate(conds):
                r = Runtime(f"sampling batch {1 + i}/{len(conds)} with {len(batched_cond)} entries").start()
                samples = self._sample(size=len(batched_cond), cond=batched_cond, **kwargs)
                samples = np.concatenate([batched_cond, samples], axis=1)
                results.append(samples)
                r.stop().print()
        r = np.concatenate(results, axis=0)
        return r

    def prob(self, xs: TTensor, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        if self.tfd_distribution is None:
            self.create_base_distribution()
        return self.likelihoods(xs, cond, batch_size)

    def log_prob(self, xs: TTensor, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        if self.tfd_distribution is None:
            self.create_base_distribution()
        return self.log_likelihoods(xs, cond, batch_size)

    def log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        if self.tfd_distribution is None:
            self.create_base_distribution()
        xs, cond = self.extract_xs_cond(xs, cond)
        xs = self.cast_2_input(xs, event_dim=self.input_dim)
        cond = self.cast_2_input(cond, event_dim=self.conditional_dims)
        return self.cast_2_likelihood(xs, self.batch_call(self._log_likelihoods, xs, cond, batch_size))

    def sample(self, size: int = 1, cond: TTensorOpt = None, batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        if self.tfd_distribution is None:
            self.create_base_distribution()
        cond = self.cast_2_input(cond, event_dim=self.conditional_dims)
        return self.batch_call_sample(size=size, cond=cond, batch_size=batch_size, **kwargs)

    @staticmethod
    def static_sample(js: str, size: int, cond: Optional[np.ndarray] = None):
        setproctitle.setproctitle('static sample')
        d: Distribution = jsonloader.from_json(js)
        return d.sample(size, cond=cond)

    def sample_in_process(self, size: int = 1, cond=None) -> np.ndarray:
        # return BaseMethods.call_func_in_process(self, self.sample, arguments={"size": size, 'cond': cond})
        js = self.to_json()
        return Global.POOL().run_blocking(Distribution.static_sample, args=(js, size, cond))

    def likelihoods(self, xs: TTensor, cond: TTensorOpt = None, batch_size: Optional[int] = None) -> np.ndarray:
        if self.tfd_distribution is None:
            self.create_base_distribution()
        xs, cond = self.extract_xs_cond(xs, cond)
        xs = self.cast_2_input(xs, event_dim=self.input_dim)
        cond = self.cast_2_input(cond, event_dim=self.conditional_dims)
        if self.conditional and self.conditional_producer_function is not None:
            d: Distribution = self.conditional_producer_function(self, cond)
            return d.batch_call(d._likelihoods, xs, cond, batch_size)
        return self.batch_call(self._likelihoods, xs, cond, batch_size)

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        raise NotImplementedError()

    def _sample(self, size: int = 1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        raise NotImplementedError()

    @property
    def heatmap_creator(self) -> HeatmapCreator:
        hm = HeatmapCreator(self)
        return hm


class CutThroughData:
    def __init__(self, f_name: str, dimension_index: int, mesh_count: int, ps: np.ndarray, us: np.ndarray, det: np.ndarray, xs: np.ndarray,
                 start: float = -4.0, end: float = 4.0, pre_title: str = "",
                 title_transform: str = "transform title", title_cut: str = "title cut", title_cut_raw: str = "title cut raw",
                 columns: Optional[List[str]] = NotProvided(), x_label: str = "x", y_label: str = "Z"):
        self.dimension_index: int = dimension_index
        self.mesh_count: int = mesh_count
        self.start: float = NotProvided.value_if_not_provided(start, -4.0)
        self.end: float = NotProvided.value_if_not_provided(end, 4.0)
        self.title_transform: str = title_transform
        self.title_cut: str = title_cut
        self.title_cut_raw: str = title_cut_raw
        self.columns: Optional[List[str]] = NotProvided.value_if_not_provided(columns, None)
        self.ps: np.ndarray = ps
        self.us: np.ndarray = us
        self.det: np.ndarray = det
        self.x_label: str = x_label
        self.y_label: str = y_label
        self.f_name: str = f_name
        self.pre_title: str = pre_title
        self.xs: np.ndarray = xs

    def print_cut_ps(self, ax_cut):
        xx = np.column_stack([self.xs, self.ps])
        df: pd.DataFrame = pd.DataFrame(xx, columns=[self.x_label, self.y_label])
        ax_cut.set_title(f"{self.y_label} = p(f{self.f_name}) * |det(df{self.f_name}/d{self.x_label})|")
        sns.lineplot(data=df, x=self.x_label, y=self.y_label, ax=ax_cut)

    def print_cut_ps_raw(self, ax_cut, use_pre_title: bool = False):
        # todo computation should happen before creating this CutThroughData object
        ps = multivariate_normal.pdf(self.us, mean=[0.0, 0.0], cov=[1.0, 1.0])
        xx = np.column_stack([self.xs, ps])
        df: pd.DataFrame = pd.DataFrame(xx, columns=[self.x_label, self.y_label])
        if use_pre_title:
            pre_title: str = f"{self.pre_title}, "
        else:
            pre_title = ""
        ax_cut.set_title(f"{pre_title}{self.y_label} = p(f{self.f_name})")
        sns.lineplot(data=df, x=self.x_label, y=self.y_label, ax=ax_cut)

    def print_transform(self, axs):
        if not axs.shape == (2,):
            raise ValueError("need 2 axs")

        def pt(ax, us: np.ndarray, y_label: str):
            xx = np.column_stack([self.xs, us])
            df: pd.DataFrame = pd.DataFrame(xx, columns=[self.x_label, y_label])
            sns.lineplot(data=df, x=self.x_label, y=y_label, ax=ax)

        fx = f"u_x{self.f_name}"
        fy = f"u_y{self.f_name}"
        axs[0].set_title(f"{self.pre_title} {self.title_transform}")
        pt(axs[0], us=self.us[:, 0], y_label=fx)
        pt(axs[1], us=self.us[:, 1], y_label=fy)
        # xx = np.column_stack([self.xs, self.ps])
        # df: pd.DataFrame = pd.DataFrame(xx, columns=[self.x_label, self.y_label])
        # sns.lineplot(data=df, x=self.x_label, y=self.y_label, ax=ax)


class HeatmapCreator(Ser):
    def __init__(self, dist: Distribution = NotProvided()):
        super().__init__()
        self.dist: Distribution = dist

    def heatmap_1d_data(self, xmin=-4.0, xmax=4.0, ymin=None, ymax=None, mesh_count=200, suptitle: str = None,
                        title: str = None, columns: Optional[List[str]] = NotProvided()) -> DensityPlotData:
        x = tf.linspace(xmin, xmax, mesh_count)
        x = cast_to_ndarray(x)
        x = x.reshape((mesh_count, 1))
        probs = self.dist.prob(x)
        values = np.concatenate([x, probs], axis=1)
        data = DensityPlotData(values=values, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, mesh_count=mesh_count, type="1d", suptitle=suptitle, title=title, columns=columns)
        return data

    def heatmap_2d_data(self, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=200, suptitle: str = None, title: str = None,
                        columns: Optional[List[str]] = NotProvided(), true_distribution: Optional[Distribution] = None) -> DensityPlotData:
        """plot_data.append(heatmap_2d_data(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, suptitle="p(x) sampled from Z", title=f"after {i} epochs"))"""
        x = tf.linspace(xmin, xmax, mesh_count)
        y = tf.linspace(ymin, ymax, mesh_count)
        X, Y = tf.meshgrid(x, y)

        concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
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
        return data

    def heatmap_2d_data(self, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=200, suptitle: str = None, title: str = None,
                        columns: Optional[List[str]] = NotProvided(), true_distribution: Optional[Distribution] = None) -> DensityPlotData:
        """plot_data.append(heatmap_2d_data(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, suptitle="p(x) sampled from Z", title=f"after {i} epochs"))"""
        x = tf.linspace(xmin, xmax, mesh_count)
        y = tf.linspace(ymin, ymax, mesh_count)
        X, Y = tf.meshgrid(x, y)

        concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
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
        return data

    def diff_2d(self, true_plot_data: DensityPlotData, title: Optional[str] = None) -> DensityPlotData:
        x = tf.linspace(true_plot_data.xmin, true_plot_data.xmax, true_plot_data.mesh_count)
        y = tf.linspace(true_plot_data.ymin, true_plot_data.ymax, true_plot_data.mesh_count)
        X, Y = tf.meshgrid(x, y)

        concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
        prob = self.dist.prob(concatenated_mesh_coordinates, batch_size=10000)
        probs = tf.reshape(prob, (true_plot_data.mesh_count, true_plot_data.mesh_count))
        probs = probs.numpy()
        # probs = np.flip(probs,axis=(1,0))
        probs = np.rot90(probs)
        # values = np.abs(true_plot_data.values - probs)
        values = probs - true_plot_data.values
        data = DensityPlotData(values=values,
                               xmin=true_plot_data.xmin,
                               xmax=true_plot_data.xmax,
                               ymin=true_plot_data.ymin,
                               ymax=true_plot_data.ymax,
                               type='diff',
                               mesh_count=true_plot_data.mesh_count,
                               title=title)
        return data
        # print('asdasdasdasd')
