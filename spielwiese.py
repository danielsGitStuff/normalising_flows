import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from common.util import set_seed
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.WeightedMultimodalMultivariate import WeightedMultimodalMultivariate

if __name__ == '__main__':
    set_seed(45)
    g1 = GaussianMultivariate(input_dim=1, mus=[-2], cov=[.1])
    g2 = GaussianMultivariate(input_dim=1, mus=[0], cov=[1])
    g3 = GaussianMultivariate(input_dim=1, mus=[2], cov=[.2])
    data_distr = WeightedMultimodalMultivariate(input_dim=1)
    data_distr.add_d(g1, weight=1)
    data_distr.add_d(g2, weight=1)
    data_distr.add_d(g3, weight=2)

    xs = data_distr.sample(1000)
    ds_samples: pd.DataFrame = pd.DataFrame(xs[:, 0], columns=['x'])
    x_min = min(xs)
    x_max = max(xs)

    xs_ps = np.linspace(start=x_min, stop=x_max, num=1000)
    ps = data_distr.prob(xs_ps)
    ps_samples = data_distr.prob(xs)

    zero_ys = np.zeros((len(xs),)) + np.random.uniform(-.01, -0.01, size=len(xs))
    zero_xs = np.column_stack([xs.flatten(), zero_ys, ps_samples.flatten()])
    df_scatter = pd.DataFrame(zero_xs, columns=['x', 'y', 'p'])
    df_ps = pd.DataFrame(np.column_stack([xs_ps, ps]), columns=['x', 'y'])

    ORANGE = '#FF7F0E'
    orange = '#ff7f0e'
    ORANGE_PALE = '#ffc490'
    BLUE = '#2D7FB7'
    TRANSPARENT = True

    ### HISTOGRAM

    fig, ax = plt.subplots(1)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    # sns.distplot(ds_samples['x'], ax=ax, bins=20, color='purple')
    sns.histplot(data=ds_samples, x='x', ax=ax, legend=False, bins=20, stat='density', color=ORANGE_PALE)
    sns.lineplot(data=df_ps, ax=ax, x='x', y='y', color='black', linewidth=2.5)
    sns.scatterplot(data=df_scatter, ax=ax, x='x', y='y', size='p', legend=False, marker="+", color=BLUE)

    plt.tight_layout()
    plt.savefig('asd_histogram.png', transparent=TRANSPARENT)
    plt.clf()

    ### NOISY HISTOGRAM

    fig, ax = plt.subplots(1)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    sns.histplot(data=ds_samples, x='x', ax=ax, legend=False, bins=200, stat='density', color=ORANGE_PALE)
    sns.lineplot(data=df_ps, ax=ax, x='x', y='y', color='black', linewidth=2.5)
    sns.scatterplot(data=df_scatter, ax=ax, x='x', y='y', size='p', legend=False, marker="+", color=BLUE)

    plt.tight_layout()
    plt.savefig('asd_histogram.noisy.png', transparent=TRANSPARENT)
    plt.clf()

    ### KDE

    fig, ax = plt.subplots(1)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    sns.kdeplot(data=ds_samples, x='x', ax=ax, legend=False, bw_adjust=.2, color=ORANGE, linewidth=2.5)
    sns.lineplot(data=df_ps, ax=ax, x='x', y='y', color='black', linewidth=2.5)
    sns.scatterplot(data=df_scatter, ax=ax, x='x', y='y', size='p', legend=False, marker="+")

    plt.tight_layout()
    plt.savefig('asd_kde.png', transparent=TRANSPARENT)
    plt.clf()

    ### KDE Noisy

    fig, ax = plt.subplots(1)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    sns.kdeplot(data=ds_samples, x='x', ax=ax, legend=False, bw_adjust=.02, color=ORANGE, linewidth=2.5)
    sns.lineplot(data=df_ps, ax=ax, x='x', y='y', color='black', linewidth=2.5)
    sns.scatterplot(data=df_scatter, ax=ax, x='x', y='y', size='p', legend=False, marker="+")

    plt.tight_layout()
    plt.savefig('asd_kde.noisy.png', transparent=TRANSPARENT)
    plt.clf()

    fig, ax = plt.subplots(1)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    ### Data, Raw

    sns.lineplot(data=df_ps, ax=ax, x='x', y='y', color='black', linewidth=2.5)
    sns.scatterplot(data=df_scatter, ax=ax, x='x', y='y', size='p', legend=False, marker="+")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig('asd_data_distribution.png', transparent=TRANSPARENT)
    plt.clf()
