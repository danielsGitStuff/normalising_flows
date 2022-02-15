from pathlib import Path

from matplotlib import pyplot as plt
from tensorflow_probability import math
from typing import Optional, List, Union
import math
import numpy as np
from tensorflow_probability.python.bijectors import Chain

from common.NotProvided import NotProvided
from common.globals import Global
from distributions.Distribution import Distribution, DensityPlotData, HeatmapCreator, TfpD
from distributions.LearnedDistribution import LearnedDistribution
from distributions.UniformMultivariate import UniformMultivariate
from distributions.base import TTensor, TTensorOpt, enable_memory_growth
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.StaticMethods import StaticMethods
import seaborn as sns
from tensorflow_probability.python.bijectors import MaskedAutoregressiveFlow as MMFF


class ChainDistribution(Distribution):
    def _create_base_distribution(self) -> Optional[TfpD]:
        pass

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        pass

    def _sample(self, size: int = 1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        pass

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        pass

    def __init__(self):
        super().__init__()
        self.chain: Chain = None


class TransformationPrinter:
    def __init__(self):
        self.denses: List[DensityPlotData] = []
        self.log_scale = False
        self.fig = None
        self.print_3d_for_denses = False
        self.name = 'NF2D_1Rect_l2'
        self.result_folder = Path('result_spielwiese')
        self.result_folder.mkdir(exist_ok=True)
        self.h_offset = 0
        self.src_distr: UniformMultivariate = UniformMultivariate(input_dim=2, lows=[-1, -2], highs=[1, 2])

    def hm(self, dist: Distribution, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=200, suptitle: str = None, title: str = None, vmax: Optional[Union[float, str]] = None,
           columns: Optional[List[str]] = NotProvided(), true_distribution: Optional[Distribution] = None):
        if self.denses is None:
            self.denses = []
        if dist.input_dim == 1:
            dp = dist.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, mesh_count=mesh_count, suptitle=suptitle, title=title, columns=columns)
        elif dist.input_dim == 2:
            dp: DensityPlotData = dist.heatmap_creator.heatmap_2d_data(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, mesh_count=mesh_count, suptitle=suptitle, title=title,
                                                                       columns=columns, true_distribution=true_distribution)
        else:
            raise RuntimeError(f"cannot handle input_dim {dist.input_dim}")
        vmin = None
        if not self.log_scale:
            vmin = 0.0
        self.denses.append((dp, vmin, vmax))

    def print_denses(self, name: Optional[str] = None):
        if self.denses is None:
            return
        fig, axs = self.default_fig(int(math.ceil(len(self.denses) / 2)), 2)
        for ax in axs.flatten():
            ax.set_axis_off()
        for (dp, vmin, vmax), ax in zip(self.denses, axs.flatten()):
            ax.set_axis_on()
            if vmax == 'auto':
                vmax = 0
                for d, _, _ in self.denses:
                    vmax = max(vmax, d.values.max())
            dp.print_yourself(ax, vmax=vmax, vmin=vmin)
        # remove empty diagram
        # if len(axs.shape) == 2 and axs.shape[0] * axs.shape[1] > len(self.denses):
        #     axs[-1][-1].set_axis_off()
        self.save_fig(name=name)
        if self.print_3d_for_denses and not Global.Testing.has('kaleido_missing_hack'):
            for dp, vmin, vmax in self.denses[1:]:
                dp.print_yourself_3d(title=dp.title, image_base_path=self.get_base_path(f"{self.name}.{dp.title}"))
        self.denses = None

    def get_base_path(self, name: Optional[str] = None) -> str:
        if name is None:
            name = self.name
        target = Path(self.result_folder, name)
        return str(target)

    def save_fig(self, transparent: bool = True, tight_layout: bool = True, name: str = NotProvided()):
        if tight_layout:
            self.fig.tight_layout()
        name: str = NotProvided.value_if_not_provided(name, self.name)
        target = f"{self.get_base_path(name)}.png"
        self.fig.savefig(target, transparent=transparent)

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

    def probe_src_dist(self, xmin: float, xmax: float, ymin: float, ymax: float ) -> np.ndarray:
        import tensorflow as tf
        # xmin = self.src_distr.lows[0]
        # ymin = self.src_distr.lows[1]
        # xmax = self.src_distr.highs[0]
        # ymax = self.src_distr.highs[1]
        mesh_count = 20

        # xmin = self.src_distr.lows[0] - 2.0
        # ymin = self.src_distr.lows[1] - 1.0
        # xmax = self.src_distr.highs[0] + 2.0
        # ymax = self.src_distr.highs[1] + 1.0
        # mesh_count = 50

        x = tf.linspace(xmin, xmax, mesh_count)
        y = tf.linspace(ymin, ymax, mesh_count)
        X, Y = tf.meshgrid(x, y)
        concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])]))
        ar = np.array(concatenated_mesh_coordinates, dtype=np.float32)
        # ar = ar.T
        return ar

    def run(self, xmin: float, xmax: float, ymin: float, ymax: float,radius:float, target: Path):
        if LearnedDistribution.can_load_from('.cache', 'NF2D_1RectL2_l2.0.maf'):
            xmin = float(xmin)
            xmax = float(xmax)
            ymin = float(ymin)
            ymax = float(ymax)
            radius = float(radius)
            print('can load')
            maf: MaskedAutoregressiveFlow = MaskedAutoregressiveFlow.load('.cache', 'NF2D_1RectL2_l2.0.maf')

            self.hm(maf, mesh_count=100)

            chain: Chain = maf.transformed_distribution.bijector
            bs = chain.bijectors

            xs = self.probe_src_dist(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            xx = np.zeros((xs.shape[0], xs.shape[1] + 1))
            xx[:, 0:2] = xs
            xx[:, 2] = np.linspace(0, 1, len(xx))
            # xx[103][2] = 2.0
            fig, axs = StaticMethods.default_fig(3, 2, w=10, h=8)
            axs = axs.reshape((3, 2))

            def printi(ax, vs: np.ndarray, c=None, cmap='viridis', determinant: bool = False):
                ax.set_box_aspect(1)
                # im = ax.scatter(vs[:, 0], vs[:, 1], c=c, cmap=cmap)
                hue = None
                if determinant:
                    hue = c
                    c = None
                sns.scatterplot(vs[:, 0], vs[:, 1], hue=hue, c=c, cmap=cmap, ax=ax, legend='auto')
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius)
                # ax.legend()
                # fig.colorbar(im, cax=ax, orientation='horizontal')

            # from distribution
            printi(axs[0][0], xs, c=xx[:, 2])

            det = chain.inverse_log_det_jacobian(xs)
            printi(axs[0][1], xs, c=det, cmap='plasma', determinant=True)
            # axs[0][1].set_axis_off()

            # after first layer
            first_maf_index = 0
            for b in chain.bijectors:
                first_maf_index += 1
                if isinstance(b, MMFF):
                    break
            first_layer_chain = Chain(chain.bijectors[:first_maf_index])
            us = first_layer_chain.inverse(xs)
            printi(axs[1][0], us, c=xx[:, 2])

            det = first_layer_chain.inverse_log_det_jacobian(xs)
            det = np.exp(det)
            printi(axs[1][1], us, c=det, cmap='plasma', determinant=True)

            # after last layer
            us = chain.inverse(xs)
            printi(axs[2][0], us, c=xx[:, 2])

            det = chain.inverse_log_det_jacobian(xs)
            det = np.exp(det)
            printi(axs[2][1], us, c=det, cmap='plasma', determinant=True)

            fig.tight_layout()
            results_dir: Path = Path('results_visualise')
            results_dir.mkdir(exist_ok=True)
            # target: Path = Path(results_dir, 'visual.png')
            # print(f"saving to '{target.absolute()}'")
            plt.savefig(target)

            # self.print_denses()

            # xs = np.zeros((3, 2), dtype=np.float32)
            #

            print('end')


if __name__ == '__main__':
    enable_memory_growth()
    results_dir: Path = Path('results_visualise')
    results_dir.mkdir(exist_ok=True)
    TransformationPrinter().run(xmin=-1, xmax=1, ymin=-2, ymax=2, radius=4, target=Path(results_dir, 'internals1.png'))
    TransformationPrinter().run(xmin=-3, xmax=3, ymin=-3, ymax=3, radius=10, target=Path(results_dir, 'internals2.png'))
