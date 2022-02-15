from __future__ import annotations

from distributions.Distribution import DensityPlotData
from distributions.GaussianMultivariate import GaussianMultivariate
from keta.argparseer import ArgParser
from maf.stuff.MafExperiment import MafExperiment
from maf.stuff.ShiftAndScale import ShiftAndScale, ShiftAndScaleWrong


class SS2DMafExperiment(MafExperiment):
    def __init__(self):
        super().__init__("SS2DExample")

    def print_divergences(self):
        pass

    def results_dir_name(self) -> str:
        return 'results_artificial_dim2'

    def _run(self):
        no_rows = 2
        no_columns = 2
        columns_u = ['u', 'v']
        columns_x = ['x', 'y']
        fig, axs = self.default_fig(no_rows=no_rows, no_columns=no_columns)

        u_distribution: GaussianMultivariate = GaussianMultivariate(input_dim=2, mus=[0.0, 0.0], cov=[1.0, 1.0])
        data_distribution: GaussianMultivariate = GaussianMultivariate(input_dim=2, mus=[0.0, 0.0], cov=[2 ** 2, 1])

        target_data: DensityPlotData = u_distribution.heatmap_creator.heatmap_2d_data(title="source")
        max_target: float = target_data.values.max()

        transform_right: ShiftAndScale = ShiftAndScale(input_dim=2, shift=[0, 0], scale=[1 / 2, 1])
        transform_wrong: ShiftAndScale = ShiftAndScaleWrong(input_dim=2, shift=[0, 0], scale=[1 / 2, 1])

        self.hm(dist=data_distribution,title="X ~ N(0, 0; 2^2, 1)", columns=columns_x,vmax=max_target)
        self.hm(dist=u_distribution,title="U ~ N(0, 0; 1, 1)", columns=columns_u,vmax=max_target)
        self.hm(dist=transform_wrong,title="X' = Scale(U)", columns=columns_x,vmax=max_target)
        self.hm(dist=transform_right,title="X = ScaleNorm(U)", columns=columns_x,vmax=max_target)

        # data_distribution.heatmap_creator.heatmap_2d_data(title="X ~ N(0, 0; 2^2, 1)", columns=columns_x).print_yourself(axs[0][0], vmax=max_target, cmap=self.heat_map_cmap)
        # u_distribution.heatmap_creator.heatmap_2d_data(title="U ~ N(0, 0; 1, 1)", columns=columns_u).print_yourself(axs[0][1], cmap=self.heat_map_cmap)
        # transform_wrong.heatmap_creator.heatmap_2d_data(title="X' = Scale(U)", columns=columns_x).print_yourself(axs[1, 0], vmax=max_target, cmap=self.heat_map_cmap)
        # transform_right.heatmap_creator.heatmap_2d_data(title="X = ScaleNorm(U)", columns=columns_x).print_yourself(axs[1, 1], vmax=max_target, cmap=self.heat_map_cmap)
        # self.save_fig()


if __name__ == '__main__':
    ArgParser.parse()
    SS2DMafExperiment().run()
