from __future__ import annotations

import math
from typing import List, Optional

from matplotlib import pyplot as plt

from common.util import Runtime
from distributions.Distribution import Distribution
from distributions.LearnedDistribution import EarlyStop
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.kl.JS import JensenShannonDivergence
from distributions.kl.KL import KullbackLeiblerDivergence
from maf.DS import DS
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.examples.stuff.MafExperiment import MafExperiment
import numpy as np
import tensorflow as tf

import pandas as pd
import seaborn as sns


class Foursome2DMafExperiment(MafExperiment):

    def __init__(self, name: str):
        super().__init__(name)
        self.h_offset = -2
        r = Runtime("creating MAFs").start()
        self.mafs: List[MaskedAutoregressiveFlow] = self.create_mafs()
        r.stop().print()
        if len(self.mafs) == 0:
            raise RuntimeError("no MAFs!!!")
        if len(self.mafs) > 3:
            raise RuntimeError("I said FOURsome!")
        self.data_distribution: Distribution = self.create_data_distribution()
        if self.data_distribution.input_dim > 2:
            raise RuntimeError("can only deal with input_dim <= 2")
        self.xmin: float = -4.0
        self.xmax: float = 4.0
        self.ymin: float = -4.0
        self.ymax: float = 4.0
        self.vmax: Optional[float] = None
        self.no_samples: int = 80000
        self.no_val_samples: int = 2000
        self.batch_size: int = 128
        self.epochs: int = 200
        self.show_3d = False
        self.mesh_count: int = 1000
        self.meh_count_cut: int = 200
        self.divergence_half_width: Optional[float] = None
        self.divergence_step_size: Optional[float] = None
        self.print_3d_for_denses = True if self.data_distribution.input_dim == 2 else False
        self.patiences: List[int] = [10, 10, 20]

        # self.mesh_count = 200
        # self.meh_count_cut = 200

    def run(self):
        super(Foursome2DMafExperiment, self).run()
        # self.print_samples()

    def print_samples(self):
        plt.clf()
        fig, axs = self.default_fig(int(math.ceil(len(self.mafs) / 2)), 2)
        for i, maf in enumerate(self.mafs):
            xs = maf.sample(200)
            df = pd.DataFrame(xs, columns=['x', 'y'])
            ax = axs[i]
            sns.scatterplot(data=df, ax=ax, x='x', y='y')
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)
            self.save_fig(name=f"z_testing_{i}.png")

    def print_divergences(self):
        if self.divergence_half_width is None:
            print("If you want KL/JS-divergence set 'divergence_half_width' and 'divergence_step_size'")
            return
        values = []
        for maf in self.mafs:
            j = JensenShannonDivergence(p=maf, q=self.data_distribution, half_width=self.divergence_half_width, step_size=self.divergence_step_size, batch_size=self.batch_size)
            k = KullbackLeiblerDivergence(p=maf, q=self.data_distribution, half_width=self.divergence_half_width, step_size=self.divergence_step_size, batch_size=self.batch_size)
            jsd = j.calculate()
            kld = k.calculate()
            row = [maf.layers, kld, jsd]
            values.append(row)
        values = np.array(values, dtype=np.float32)
        df: pd.DataFrame = pd.DataFrame(values, columns=['layers', 'kl', 'js'])
        df_file = self.get_base_path(f"{self.name}.divergences.csv")
        df.to_csv(df_file)

    def set_minmax_square(self, minimax: [float, int]):
        maxi = abs(float(minimax))
        mini = -maxi
        self.xmin = mini
        self.ymin = mini
        self.xmax = maxi
        self.ymax = maxi

    def create_data_distribution(self) -> Distribution:
        raise NotImplementedError()

    def create_mafs(self) -> List[MaskedAutoregressiveFlow]:
        raise NotImplementedError()

    def create_data_title(self) -> str:
        raise NotImplementedError()

    def _run(self):
        xs: np.ndarray = self.data_distribution.sample(self.no_samples)
        val_xs: np.ndarray = self.data_distribution.sample(self.no_val_samples)
        ds: DS = DS.from_tensor_slices(xs)
        val_ds: DS = DS.from_tensor_slices(val_xs)
        self.hm(dist=self.data_distribution, title=self.create_data_title(), xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, vmax=self.vmax)
        mafs = []
        for i, maf in enumerate(self.mafs):
            prefix = self.maf_prefix(maf.layers)
            if LearnedTransformedDistribution.can_load_from(self.cache_dir, prefix=prefix):
                maf: MaskedAutoregressiveFlow = LearnedTransformedDistribution.load(self.cache_dir, prefix=prefix)
            else:
                es = EarlyStop(monitor="val_loss", comparison_op=tf.less, patience=self.patiences[i], restore_best_model=True)
                maf.fit(dataset=ds, batch_size=self.batch_size, epochs=self.epochs, val_xs=val_ds, early_stop=es)
                maf.save(self.cache_dir, prefix=prefix)
            mafs.append(maf)
            title = f"MAF {maf.layers}L"
            print(f"heatmap for '{title}'")
            self.hm(dist=maf, title=title, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, vmax=self.vmax, mesh_count=self.mesh_count,
                    true_distribution=self.data_distribution)
            print(f"cut for '{title}'")
            self.cut(maf, x_start=self.xmin, x_end=self.xmax, y_start=self.ymin, y_end=self.ymax, mesh_count=self.meh_count_cut, pre_title=title)
        self.mafs = mafs
        # add 3d
        # dp, _ = self.denses[-1]
        # dp: DensityPlotData = dp
        # dp.print_yourself_3d(title, show=self.show_3d, image_base_path=self.get_base_path())
        # maf.heatmap_creator.heatmap_2d_data(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, mesh_count=self.mesh_count,
        #                                     true_distribution=self.data_distribution).print_yourself_3d(
        #     f"MAF {maf.layers}L", show=True, image_base_path=self.get_base_path())
