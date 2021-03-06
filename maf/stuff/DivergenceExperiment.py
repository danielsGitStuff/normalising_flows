from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
import setproctitle
import tensorflow as tf
from matplotlib import pyplot as plt

from common import jsonloader, util
from common.NotProvided import NotProvided
from common.globals import Global
from common.jsonloader import Ser
from common.util import Runtime
from distributions.Distribution import Distribution, DensityPlotData
from distributions.LearnedDistribution import EarlyStop
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.base import enable_memory_growth, BaseMethods
from distributions.kl.DivergenceMetric import DivergenceMetric
from distributions.kl.KL import KullbackLeiblerDivergence
from maf.DS import DS
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.MafExperiment import MafExperiment
from common.prozess.Prozessor import WorkLoad


class DivergenceProcess(Ser):
    @staticmethod
    def static_run(js: str, task: str, xs, val_xs: np.ndarray, xs_samples: np.ndarray, log_ps_samples: np.ndarray) -> Tuple[Path, str]:
        dp: DivergenceProcess = jsonloader.from_json(js)
        dp.xs = xs
        dp.xs_samples = xs_samples
        dp.val_xs = val_xs
        dp.log_ps_samples = log_ps_samples
        setproctitle.setproctitle(f"{task}, MAF.fit L{dp.maf.layers}")
        return dp.run()

    def __init__(self, cache_dir: Path = NotProvided(),
                 prefix: str = NotProvided(),
                 use_early_stop: bool = NotProvided(),
                 epochs: int = NotProvided(),
                 batch_size: int = NotProvided(),
                 patience: int = NotProvided(),
                 divergence_metric_every_epoch: int = NotProvided(),
                 maf: MaskedAutoregressiveFlow = NotProvided(),
                 xs: Optional[np.ndarray] = NotProvided(),
                 val_xs: Optional[np.ndarray] = NotProvided(),
                 xs_samples: Optional[np.ndarray] = NotProvided(),
                 log_ps_samples: Optional[np.ndarray] = NotProvided()):
        super().__init__()
        self.cache_dir: Path = cache_dir
        self.maf: MaskedAutoregressiveFlow = maf
        self.xs: Optional[np.ndarray] = xs
        self.val_xs: Optional[np.ndarray] = val_xs
        self.prefix: str = prefix
        self.use_early_stop: bool = use_early_stop
        self.xs_samples: Optional[np.ndarray] = xs_samples
        self.patience: int = patience
        self.divergence_metric_every_epoch: int = divergence_metric_every_epoch
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.log_ps_samples: Optional[np.ndarray] = log_ps_samples

    def run(self) -> Tuple[Path, str]:
        if LearnedTransformedDistribution.can_load_from(self.cache_dir, prefix=self.prefix):
            # pass
            self.maf: MaskedAutoregressiveFlow = MaskedAutoregressiveFlow.load(self.cache_dir, prefix=self.prefix)
        else:
            enable_memory_growth()
            Global.set_seed(util.randomSeed())
            ds: DS = DS.from_tensor_slices(self.xs)
            val_ds: DS = DS.from_tensor_slices(self.val_xs)
            ds_samples: Optional[DS] = None
            log_ps_samples = None
            if self.xs_samples is not None:
                ds_samples = DS.from_tensor_slices(self.xs_samples)
                log_ps_samples = DS.from_tensor_slices(self.log_ps_samples)
            es = None
            if self.use_early_stop:
                es = EarlyStop(monitor="val_loss", comparison_op=tf.less, patience=self.patience, restore_best_model=True)
            divergence_metric = None
            if ds_samples is not None:
                divergence_metric = DivergenceMetric(maf=self.maf, ds_samples=ds_samples, log_ps_samples=log_ps_samples,
                                                     run_every_epoch=self.divergence_metric_every_epoch)
            self.maf.fit(dataset=ds, batch_size=self.batch_size, epochs=self.epochs, val_xs=val_ds, early_stop=es, divergence_metric=divergence_metric)
            self.maf.save(self.cache_dir, prefix=self.prefix)
        return self.cache_dir, self.prefix


class DivergenceExperiment(MafExperiment):
    def __init__(self, name: str, layers: List[int] = None, layers_repeat: int = 1, pool_size: int = 6):
        super().__init__(name, pool_size=pool_size)
        self.layers: List[int] = layers or [10, 10, 10, 20, 20, 20, 30, 30, 30]
        self.layers_repeat: int = layers_repeat
        self.xmin: float = -4.0
        self.xmax: float = 4.0
        self.ymin: float = -4.0
        self.ymax: float = 4.0
        self.vmax: Optional[float, str] = 'auto'
        self.vmax_diff: Union[str, float] = 'auto'
        self.no_samples: int = 1024 * 30
        self.no_val_samples: int = 1024 * 2
        self.mesh_count: int = 1000
        self.meh_count_cut: int = 200
        self.batch_size: int = 1024
        self.epochs: int = 2000
        r = Runtime("creating MAFs").start()
        self.mafs: List[MaskedAutoregressiveFlow] = self.create_mafs()
        util.p(f"created {len(self.mafs)} NFs")
        for i, maf in enumerate(self.mafs):
            util.p(f"{i}: {maf.layers} layers")
        r.stop().print()
        # todo deprecate sampling space
        self.divergence_half_width: Optional[float] = None
        self.divergence_step_size: Optional[float] = None
        self.divergence_sample_size: Optional[int] = 10000
        self.data_distribution: Distribution = self.create_data_distribution()
        self.divergence_metric_every_epoch: int = 25
        self.patiences: List[int] = [50] * len(self.mafs)

        self.xs_samples: Optional[np.ndarray] = None
        self.log_ps_samples: Optional[np.ndarray] = None

    def get_layers(self) -> List[int]:
        return list(reversed(sorted(self.layers * self.layers_repeat)))

    def set_minmax_square(self, minimax: [float, int]):
        maxi = abs(float(minimax))
        mini = -maxi
        self.xmin = mini
        self.ymin = mini
        self.xmax = maxi
        self.ymax = maxi

    def _print_datadistribution(self):
        plt.clf()
        if self.data_distribution.input_dim == 2:
            self.denses = []
            print('printing original distribution')
            args = dict(xmin=-10, xmax=10, ymin=-10, ymax=10, mesh_count=200, title=f"distribution")
            dp: DensityPlotData = BaseMethods.call_func_in_process(self.data_distribution.heatmap_creator, f=self.data_distribution.heatmap_creator.heatmap_2d_data, arguments=args)
            self.denses.append((dp, None, dp.values.max()))
            self.print_denses(name=f"{self.name}_data")
            self.denses = []

    def _print_dataset(self, xs: np.ndarray = None, suffix: str = ""):
        plt.clf()
        if len(suffix) > 0 and not suffix.startswith('_'):
            suffix = f"_{suffix}"
        if self.data_distribution.input_dim == 2:
            # plt.scatter(xs[:, 0], xs[:, 1])
            plt.figure(figsize=(10, 10))
            sns.scatterplot(x=xs[:, 0], y=xs[:, 1])
            plt.ylim(self.ymin, self.ymax)
            plt.xlim(self.xmin, self.xmax)
            plt.savefig(self.get_base_path(f"{self.name}_samples{suffix}"))

    def create_data_distribution(self) -> Distribution:
        raise NotImplementedError()

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        raise NotImplementedError()

    def create_data_title(self) -> str:
        raise NotImplementedError()

    def _run(self):
        xs: np.ndarray = self.data_distribution.sample_in_process(self.no_samples)
        val_xs: np.ndarray = self.data_distribution.sample_in_process(self.no_val_samples)
        self._print_datadistribution()
        self._print_dataset(xs=xs, suffix="xs")
        self._print_dataset(xs=val_xs, suffix="xs_val")

        mafs = []
        if self.divergence_metric_every_epoch > 0:
            self.xs_samples = self.data_distribution.sample_in_process(self.divergence_sample_size)
            self.log_ps_samples = BaseMethods.call_func_in_process(self.data_distribution, self.data_distribution.log_prob, arguments=[self.xs_samples])
        for i, maf in enumerate(self.mafs):
            prefix = self.maf_prefix(f"l{maf.layers}.{i}")
            dp: DivergenceProcess = DivergenceProcess(cache_dir=self.cache_dir,
                                                      prefix=prefix,
                                                      use_early_stop=self.use_early_stop,
                                                      epochs=self.epochs,
                                                      batch_size=self.batch_size,
                                                      patience=self.patiences[i],
                                                      divergence_metric_every_epoch=self.divergence_metric_every_epoch,
                                                      maf=maf,
                                                      xs=None,
                                                      val_xs=None,
                                                      xs_samples=None,
                                                      log_ps_samples=None)
            js = dp.to_json()
            # cache_d, pre = DivergenceProcess.static_run(js)
            # m: MaskedAutoregressiveFlow = MaskedAutoregressiveFlow.load(cache_d, pre)
            # mafs.append(m)
            # DivergenceProcess.static_run(js, self.name, xs, val_xs, self.xs_samples, self.log_ps_samples)
            self.prozessor.run_later(WorkLoad.create_static_method_workload(DivergenceProcess.static_run, args=(js, self.name, xs, val_xs, self.xs_samples, self.log_ps_samples)))
            # self.pool.apply_async(DivergenceProcess.static_run, args=(js, self.name, xs, val_xs, self.xs_samples, self.log_ps_samples))
        # results: List[Tuple[Path, str]] = self.pool.join()
        results: List[Tuple[Path, str]] = self.prozessor.join()
        mafs = [MaskedAutoregressiveFlow.load(cache, prefix) for cache, prefix in results]
        if self.data_distribution.input_dim < 3:
            enable_memory_growth()
            self.original_plot_data = self.hm(dist=self.data_distribution, title=self.create_data_title(), xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax,
                                              vmax=self.vmax,
                                              mesh_count=self.mesh_count)
        for maf in mafs:
            if self.data_distribution.input_dim < 3:
                title = f"MAF {maf.layers}L"
                print(f"heatmap/diff for '{title}'")
                self.diff(dist=maf, unique_property='layers', title=title, vmax=self.vmax_diff)
                self.hm(dist=maf, title=title, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, vmax=self.vmax, mesh_count=self.mesh_count,
                        true_distribution=self.data_distribution)
                # print(f"cut for '{title}'")
                # self.cut(maf, x_start=self.xmin, x_end=self.xmax, y_start=self.ymin, y_end=self.ymax, mesh_count=self.meh_count_cut, pre_title=title)
        self.mafs = mafs

    def print_divergences(self):
        if (self.divergence_half_width is None or self.divergence_step_size is None) and self.divergence_sample_size is None:
            print("If you want KL/JS-divergence set either 'divergence_half_width' and 'divergence_step_size, or 'divergence_sample_size''")
            return
        if self.divergence_sample_size is not None and (self.divergence_step_size is not None or self.divergence_half_width is not None):
            print(f"You defined 'divergence_half_width' or 'divergence_step_size' and additionally 'divergence_sample_size'. Set the first two only or the latter.",
                  file=sys.stderr)
            return
        values = []
        enable_memory_growth()
        ds_samples = DS.from_tensor_slices(self.xs_samples)
        log_ps_samples = DS.from_tensor_slices(self.log_ps_samples)
        for maf in self.mafs:
            # j = JensenShannonDivergence(p=maf, q=self.data_distribution, half_width=self.divergence_half_width, step_size=self.divergence_step_size, batch_size=self.batch_size)
            # k = KullbackLeiblerDivergence(p=maf, q=self.data_distribution, half_width=self.divergence_half_width, step_size=self.divergence_step_size, batch_size=self.batch_size)
            # j = JensenShannonDivergence(q=maf, p=self.data_distribution, half_width=self.divergence_half_width, step_size=self.divergence_step_size, batch_size=self.batch_size)
            k = KullbackLeiblerDivergence(q=maf, p=self.data_distribution, half_width=self.divergence_half_width, step_size=self.divergence_step_size, batch_size=self.batch_size)
            if self.xs_samples is None:
                # jsd = j.calculate_by_sampling_p(self.divergence_sample_size) if self.divergence_sample_size is not None else j.calculate_by_sampling_space()
                kld = k.calculate_by_sampling_p(self.divergence_sample_size) if self.divergence_sample_size is not None else k.calculate_by_sampling_space()
            else:
                # jsd = j.calculate_from_samples_vs_q(ds_p_samples=self.ds_samples, log_p_samples=self.log_ps_samples)
                kld = k.calculate_from_samples_vs_q(ds_p_samples=ds_samples, log_p_samples=log_ps_samples)
            # row = [maf.layers, kld, jsd]
            row = [maf.layers, kld]
            values.append(row)
        values = np.array(values, dtype=np.float32)
        # df: pd.DataFrame = pd.DataFrame(values, columns=['layers', 'kl', 'js'])
        df: pd.DataFrame = pd.DataFrame(values, columns=['layers', 'kl'])
        df_file = self.get_base_path(f"{self.name}.divergences.csv")
        df.to_csv(df_file)
