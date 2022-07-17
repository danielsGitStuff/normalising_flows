import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

from common.globals import Global
from distributions.distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from common.argparser import ArgParser
from maf.stuff.MafExperiment import MafExperiment
from maf.stuff.ShiftAndScale import ShiftAndScale, ShiftAndScaleWrong
import seaborn as sns
import pandas as pd


class SS1DMafExperiment(MafExperiment):
    def __init__(self):
        super().__init__("SS1DExample")

    def print_divergences(self):
        pass

    def results_dir_name(self) -> str:
        return 'results_artificial_dim1'

    def print_denses(self, name: Optional[str] = None, scatter: bool = False, samples_list: Optional[List[np.ndarray]] = None):
        if self.denses is None:
            return
        if samples_list is None:
            samples_list = [None] * len(self.denses)
        fig, axs = self.default_fig(int(math.ceil(len(self.denses) / 2)), 2)
        for ax in axs.flatten():
            ax.set_axis_off()
        for (dp, vmin, vmax), ax, samples in zip(self.denses, axs.flatten(), samples_list):
            ax.set_axis_on()
            if vmax == 'auto':
                vmax = 0
                for d, _, _ in self.denses:
                    vmax = max(vmax, d.values.max())
            dp.print_yourself(ax, vmax=vmax, vmin=vmin, scatter_too=scatter)
            if samples is not None:
                # df_samples: pd.DataFrame = pd.DataFrame(samples, columns=['x', 'y'])
                # sns.barplot(data=df_samples)
                offset = 0.00001
                BLUE_U = '#8AD4E4'
                for x, p in samples:
                    # print(f"x {x} -> {p}")
                    df: pd.DataFrame = pd.DataFrame([[x, 0.0], [x + offset, p]], columns=['x', 'p'])
                    sns.lineplot(data=df, x='x', y='p', legend=False, linewidth=1.9, color=BLUE_U, ax=ax)

        plt.tight_layout()
        self.save_fig(name=name)
        if self.print_3d_for_denses and not Global.Testing.has('kaleido_missing_hack'):
            for dp, vmin, vmax in self.denses[1:]:
                dp.print_yourself_3d(title=dp.title, image_base_path=self.get_base_path(f"{self.name}.{dp.title}"))
        self.denses = None



    def _run(self):
        fig, axs = self.default_fig(1, 2)
        columns_x = ['x', 'p(x)']
        columns_u = ['u', 'q(u)']
        xmin = -6.0
        xmax = 6.0
        data_distribution: Distribution = GaussianMultivariate(input_dim=1, mus=[0], cov=[2 ** 2])
        u_distribution: Distribution = GaussianMultivariate(input_dim=1, mus=[0], cov=[1])
        transform_right: ShiftAndScale = ShiftAndScale(shift=0, scale=1 / 2, input_dim=1)
        transform_wrong: ShiftAndScale = ShiftAndScaleWrong(shift=0, scale=1 / 2, input_dim=1)
        transform_nothing: ShiftAndScale = ShiftAndScaleWrong(shift=0, scale=1, input_dim=1)
        line1 = "p(x) = X ~ N(0;2^2)"
        line2 = "q(u) = U ~ N(0;1)"
        line3 = "p(x) = q(x/2)"
        line4 = "p(x) = q(x/2)/2"

        u_samples = u_distribution.sample(10)
        u_ps = u_distribution.prob(u_samples)

        samples_u_original = np.column_stack([u_samples, u_ps])

        samples_u_scaled = np.column_stack([u_samples * 2, u_ps])

        samples_x = np.column_stack([u_samples * 2, u_ps / 2])

        self.hm(dist=data_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2)
        self.print_denses(name="SS1DExample_just_2", samples_list=[None, samples_u_original])

        self.hm(dist=data_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2)
        self.hm(dist=transform_nothing, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title="p(x) = q(x)")
        self.print_denses(name="SS1DExample_just_3_1", samples_list=[None, samples_u_original, samples_u_original])

        self.hm(dist=data_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2)
        self.hm(dist=transform_wrong, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line3)
        self.print_denses(name="SS1DExample_just_3_2", samples_list=[None, samples_u_original, samples_u_scaled])

        self.hm(dist=data_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2)
        self.hm(dist=transform_wrong, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line3)
        self.hm(dist=transform_right, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line4)
        self.print_denses(samples_list=[None, samples_u_original, samples_u_scaled, samples_x])

        # data_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1).print_yourself(axs[0])
        # u_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2).print_yourself(axs[1])
        # self.save_fig(name="SS1DExample_just_2")
        #
        # fig, axs = self.default_fig(2, 2)
        # data_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1).print_yourself(axs[0][0])
        # u_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2).print_yourself(axs[0][1])
        # transform_nothing.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title="p(x) = q(x)").print_yourself(axs[1][0])
        # axs[1][1].set_axis_off()
        # self.save_fig(name="SS1DExample_just_3_1")
        #
        # fig, axs = self.default_fig(2, 2)
        # data_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1).print_yourself(axs[0][0])
        # u_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2).print_yourself(axs[0][1])
        # transform_wrong.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line3).print_yourself(axs[1][0])
        # axs[1][1].set_axis_off()
        # self.save_fig(name="SS1DExample_just_3_2")
        #
        # fig, axs = self.default_fig(2, 2)
        # data_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1).print_yourself(axs[0][0])
        # u_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2).print_yourself(axs[0][1])
        # transform_wrong.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line3).print_yourself(axs[1][0])
        # transform_right.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line4).print_yourself(axs[1][1])
        # self.save_fig()


if __name__ == '__main__':
    ArgParser.parse()
    SS1DMafExperiment().run()
