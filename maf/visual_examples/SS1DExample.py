from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from maf.stuff.MafExperiment import MafExperiment
from maf.stuff.ShiftAndScale import ShiftAndScale, ShiftAndScaleWrong


class SS1DMafExperiment(MafExperiment):
    def __init__(self):
        super().__init__("SS1DExample")

    def print_divergences(self):
        pass

    def results_dir_name(self) -> str:
        return 'results_artificial'

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

        self.hm(dist=data_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2 )
        self.print_denses(name="SS1DExample_just_2")


        self.hm(dist=data_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2 )
        self.hm(dist=transform_nothing,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title="p(x) = q(x)")
        self.print_denses(name="SS1DExample_just_3_1")


        self.hm(dist=data_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2 )
        self.hm(dist=transform_wrong,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line3)
        self.print_denses(name="SS1DExample_just_3_2")

        self.hm(dist=data_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line1)
        self.hm(dist=u_distribution,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title=line2 )
        self.hm(dist=transform_wrong,xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line3)
        self.hm(dist=transform_right, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title=line4)
        self.print_denses()


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
    SS1DMafExperiment().run()
