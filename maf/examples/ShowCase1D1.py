from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from maf.examples.stuff.MafExperiment import MafExperiment


class ShowCase1D1(MafExperiment):
    def __init__(self):
        super().__init__("ShowCase1D1")

    def _run(self):
        fig, axs = self.default_fig(1, 2)
        columns_x = ['x', 'p(x)']
        columns_u = ['u', 'q(u)']
        xmin = -6.0
        xmax = 6.0
        P1 = 2.6
        V1 = 0.8
        data_distribution: Distribution = MultimodalDistribution(input_dim=1, distributions=[GaussianMultivariate(input_dim=1, mus=[-P1], cov=[V1]),
                                                                                             GaussianMultivariate(input_dim=1, mus=[P1], cov=[V1]),

                                                                                             GaussianMultivariate(input_dim=1, mus=[-.9], cov=[0.1]),
                                                                                             GaussianMultivariate(input_dim=1, mus=[.9], cov=[0.1]),

                                                                                             GaussianMultivariate(input_dim=1, mus=[-.35], cov=[0.1]),
                                                                                             GaussianMultivariate(input_dim=1, mus=[.35], cov=[0.1]),

                                                                                             GaussianMultivariate(input_dim=1, mus=[-.4], cov=[0.4]),
                                                                                             GaussianMultivariate(input_dim=1, mus=[0], cov=[0.4]),
                                                                                             GaussianMultivariate(input_dim=1, mus=[.4], cov=[0.4])])
        u_distribution: Distribution = GaussianMultivariate(input_dim=1, mus=[0], cov=[1])
        self.hm(dist=data_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title="X ~ Arbitrary")
        self.hm(dist=u_distribution, xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title="U ~ N(0;1)")
        # data_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_x, title="X ~ Arbitrary").print_yourself(axs[0])
        # u_distribution.heatmap_creator.heatmap_1d_data(xmin=xmin, xmax=xmax, ymin=0.0, ymax=0.5, columns=columns_u, title="U ~ N(0;1)").print_yourself(axs[1])

        # self.save_fig(tight_layout=False)


if __name__ == '__main__':
    ShowCase1D1().run()
