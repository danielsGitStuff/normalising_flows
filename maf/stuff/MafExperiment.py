import sys
from numbers import Number

from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from distributions.distribution import CutThroughData, Distribution, ConditionalDensityPlotData
from distributions.density_plot_data import DensityPlotData
from maf.stuff.StaticMethods import StaticMethods
from matplotlib.colors import Colormap, TwoSlopeNorm
from typing import List, Tuple, Optional, Union, Dict, Set, Any

from common.NotProvided import NotProvided
from common.globals import Global
from common.util import Runtime
from maf import MaskedAutoregressiveFlow
from common.prozess.Prozessor import Prozessor


class MafExperiment:
    def __init__(self, name: str, pool_size: int = 6):
        self.cache_dir: Path = StaticMethods.cache_dir()
        self.name: str = name
        self.result_folder: Path = Global.get_default('results_dir', Path(self.results_dir_name()))
        self.heat_map_cmap: Colormap = sns.color_palette("Blues", as_cmap=True)
        self.heat_map_cmap = None
        self.fig = None
        self.cuts: Optional[List[Tuple[CutThroughData, CutThroughData]]] = None
        self.denses: Optional[List[Tuple[ConditionalDensityPlotData, Optional[float], Optional[Union[float, str]]]]] = None
        self.differences: Optional[List[Tuple[ConditionalDensityPlotData, Optional[float], Optional[Union[float, str]]]]] = None
        self.differences_keys: Set[str] = set()
        self.overall_runtime: Runtime = Runtime(f"overall runtime for '{self.name}'").start()
        self.print_3d_for_denses: bool = False
        self.log_scale: bool = False
        self.h_offset: int = 0
        self.use_early_stop: bool = True
        plt.clf()
        self.pool_size = pool_size
        # self.pool: RestartingPoolReplacement = RestartingPoolReplacement(self.pool_size)
        self.prozessor: Prozessor = Prozessor(max_workers=self.pool_size)
        self.original_plot_data: Optional[ConditionalDensityPlotData] = None
        # self.cmap_divergence = sns.diverging_palette(220, 20, as_cmap=True)
        self.cmap_divergence = sns.color_palette("icefire", as_cmap=True)
        self.cmap_absolute = sns.color_palette("rocket", as_cmap=True)
        self.conditional_values: Optional[List[List[Number]]] = None

    def set_pool_size(self, size: int):
        self.pool_size = size
        # self.pool.close()
        # self.pool: RestartingPoolReplacement = RestartingPoolReplacement(self.pool_size)
        self.prozessor.close()
        self.prozessor: Prozessor = Prozessor(max_workers=self.pool_size)

    def results_dir_name(self) -> str:
        return 'results'

    def default_fig(self, no_rows: int, no_columns: int):
        plt.clf()
        # following line ignores xtick, ytick in 1x2 subplots but not 2x2, remove  StaticMethods.default_fig() at the end of this method
        # fig, axs = StaticMethods.default_fig(no_rows=no_rows, no_columns=no_columns, h=10, w=8)
        big = 32
        medium = 28
        small = 24
        plt.rc('font', size=small)
        plt.rc('axes', titlesize=small)
        plt.rc('axes', labelsize=medium)
        plt.rc('xtick', labelsize=small)
        plt.rc('ytick', labelsize=small)
        plt.rc('legend', fontsize=small)
        plt.rc('figure', titlesize=big)
        plt.rc('lines', linewidth=3)
        self.fig, axs = StaticMethods.default_fig(no_rows=no_rows, no_columns=no_columns, h=10, w=10, h_offset=self.h_offset)
        return self.fig, axs

    def cut(self, maf: MaskedAutoregressiveFlow, pre_title: str, x_start: float = -4.0, x_end: float = 4.0, y_start: float = -4.0, y_end: float = 4.0, mesh_count: int = 200):
        if maf.input_dim != 2:
            return
        if self.cuts is None:
            self.cuts = []
        self.cuts.append((maf.heatmap_creator.cut_along_x(x_start=x_start, x_end=x_end, mesh_count=mesh_count, pre_title=pre_title),
                          maf.heatmap_creator.cut_along_y(y_start=y_start, y_end=y_end, mesh_count=mesh_count, pre_title=pre_title)))

    def diff(self, dist: Distribution, unique_property: Optional[str] = None, title: Optional[str] = None, vmax: Optional[Union[float, str]] = None):
        if dist.input_dim > 2 or dist.input_dim < 1:
            return
        if self.original_plot_data is None:
            raise RuntimeError('no original plot data. diff is impossible')
        if self.differences is None:
            self.differences = []
        if unique_property is not None:
            key: Any = getattr(dist, unique_property)
            if key in self.differences_keys:
                return
            dp = dist.heatmap_creator(conditional_values=self.conditional_values).diff_2d(self.original_plot_data, title=title)
            vmin = None
            self.differences.append((dp, vmin, vmax))
            self.differences_keys.add(key)

    def hm(self, dist: Distribution, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=200, suptitle: str = None, title: str = None, vmax: Optional[Union[float, str]] = None,
           columns: Optional[List[str]] = NotProvided(), true_distribution: Optional[Distribution] = None) -> DensityPlotData:
        if self.denses is None:
            self.denses = []
        if dist.input_dim == 1:
            dp = dist.heatmap_creator(conditional_values=self.conditional_values).heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, mesh_count=mesh_count,
                                                                                                  suptitle=suptitle, title=title, columns=columns)
        elif dist.input_dim == 2:
            dp: ConditionalDensityPlotData = dist.heatmap_creator(conditional_values=self.conditional_values).heatmap_2d_data(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                                                                                                              mesh_count=mesh_count, suptitle=suptitle, title=title,
                                                                                                                              columns=columns, true_distribution=true_distribution)
        else:
            raise RuntimeError(f"cannot handle input_dim {dist.input_dim}")
        vmin = None
        if not self.log_scale:
            vmin = 0.0
        self.denses.append((dp, vmin, vmax))
        return dp

    def print_denses(self, name: Optional[str] = None):
        if self.denses is None:
            return
        no_of_plots = sum([d[0].plot_count() for d in self.denses])
        fig, axs = self.default_fig(int(math.ceil(no_of_plots / 2)), 2)
        for ax in axs.flatten():
            ax.set_axis_off()
        ax_index = 0
        axs = axs.flatten()
        for (cdp, vmin, vmax) in self.denses:
            cdp: ConditionalDensityPlotData = cdp
            for i, dp in enumerate(cdp.density_plots):
                dp: DensityPlotData = dp
                ax = axs[ax_index]
                ax_index += 1
                ax.set_axis_on()
                if vmax == 'auto':
                    vmax = 0
                    for cd, _, _ in self.denses:
                        cd: ConditionalDensityPlotData = cd
                        for d in cd.density_plots:
                            vmax = max(vmax, d.values.max())
                dp.print_yourself(ax, vmax=vmax, vmin=vmin, cmap=self.cmap_absolute)
        # remove empty diagram
        # if len(axs.shape) == 2 and axs.shape[0] * axs.shape[1] > len(self.denses):
        #     axs[-1][-1].set_axis_off()
        self.save_fig(name=name, transparent=True)
        if self.print_3d_for_denses and not Global.Testing.has('kaleido_missing_hack'):
            for dp, vmin, vmax in self.denses[1:]:
                dp.print_yourself_3d(title=dp.title, image_base_path=self.get_base_path(f"{self.name}.{dp.title}"))
        self.denses = None

    def print_diffs(self):
        if self.differences is None:
            return
        no_of_plots = sum([d[0].plot_count() for d in self.differences]) + self.original_plot_data.plot_count()
        fig, axs = self.default_fig(int(math.ceil((no_of_plots + 1) / 3)), 3)
        for ax in axs.flatten():
            ax.set_axis_off()
        axs = axs.flatten()
        self.original_plot_data.print_yourself(axs[0: self.original_plot_data.plot_count()], cmap=self.cmap_absolute)
        axs = axs[self.original_plot_data.plot_count():]

        # max_abs_divergence: float = max([np.abs(dp.values).max() for dp, _, _ in self.differences])
        max_abs_divergence: float = max([np.abs(dp.values).max() for cdp, _, _ in self.differences for dp in cdp.density_plots])
        divergence_norm = TwoSlopeNorm(vmin=-max_abs_divergence, vcenter=0, vmax=max_abs_divergence)

        ax_index = 0
        for (cdp, vmin, vmax) in reversed(self.differences):
            cdp: ConditionalDensityPlotData = cdp
            for dp in cdp.density_plots:
                ax = axs[ax_index]
                ax_index += 1
                ax.set_axis_on()
                if vmax == 'auto':
                    vmax = 0
                    for cd, _, _ in self.differences:
                        cd: ConditionalDensityPlotData = cd
                        for d in cd.density_plots:
                            vmax = max(vmax, d.values.max())
                legend = ax_index == 1 or True
                dp.print_yourself(ax, vmax=vmax, vmin=vmin, legend=legend, cmap=self.cmap_divergence, norm=divergence_norm)

        # for i, ((dp, vmin, vmax), ax) in enumerate(zip(reversed(self.differences), axs)):
        #     ax.set_axis_on()
        #     if vmax == 'auto':
        #         vmax = 0
        #         for d, _, _ in self.differences:
        #             vmax = max(vmax, d.values.max())
        #     legend = i == 1 or True
        #     dp.print_yourself(ax, vmax=vmax, vmin=vmin, legend=legend, cmap=self.cmap_divergence, norm=divergence_norm)
        name = f"{self.name}.diff"
        self.save_fig(name=name, transparent=True)
        self.differences = None
        self.differences_keys = set()
        # sys.exit(76)

    def print_cuts(self):
        if self.cuts is None:
            return
        fig, axs = self.default_fig(len(self.cuts * 4), 2)
        for i, tup in enumerate(self.cuts):
            c_x, c_y = tup
            c_x.print_cut_ps_raw(axs[0 + i * 4][0], use_pre_title=True)
            c_y.print_cut_ps_raw(axs[0 + i * 4][1])
            c_x.print_cut_ps(axs[1 + i * 4][0])
            c_y.print_cut_ps(axs[1 + i * 4][1])
            c_x.print_transform(axs[2 + i * 4])
            c_y.print_transform(axs[3 + i * 4])
        self.save_fig(name=f"{self.name}.cuts")
        self.cuts = None

    def run(self):
        self.result_folder.mkdir(exist_ok=True)
        # enable_memory_growth()
        self._run()
        self.print_divergences()
        self.print_diffs()
        self.print_denses()
        self.print_cuts()
        self.overall_runtime.stop().print()

    def print_divergences(self):
        raise NotImplementedError()

    def _run(self):
        raise NotImplementedError()

    def get_base_path(self, name: Optional[str] = None) -> str:
        if name is None:
            name = self.name
        target = Path(self.result_folder, name)
        return str(target)

    def save_fig(self, transparent: bool = False, tight_layout: bool = True, name: str = NotProvided()):
        if tight_layout:
            self.fig.tight_layout()
        name: str = NotProvided.value_if_not_provided(name, self.name)
        target = f"{self.get_base_path(name)}.png"
        plt.tight_layout()
        self.fig.savefig(target, transparent=transparent)

    def maf_prefix(self, optional: [str, int, float] = '') -> str:
        return f"{self.name}_{optional}.maf"
