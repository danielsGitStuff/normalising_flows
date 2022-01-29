from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Optional

import numpy as np

from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.base import enable_memory_growth
from distributions.kl.KL import KullbackLeiblerDivergence
from keta.lazymodel import LazyModel
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.stuff.MafExperiment import MafExperiment
import pandas as pd


class Plan:
    def __init__(self, dims: List[int], no_of_training_samples: List[int], no_of_layers: List[int], kl_half_width: float, kl_step_size: float, kl_batch_size: int = 100000,
                 kl_offsets: Optional[List[float]] = None):
        """@param kl_offsets: one float for each dimension. is set to zeroes otherwise"""
        self.kl_half_width: float = kl_half_width
        self.kl_step_size: float = kl_step_size
        self.kl_batch_size: int = kl_batch_size
        self.kl_offsets: List[float] = kl_offsets
        self.dims: List[int] = dims
        self.no_of_training_samples: List[int] = no_of_training_samples
        self.no_of_layers: List[int] = no_of_layers
        self.metrics: List[str] = ['kl_divergence']
        middle_block: np.ndarray = np.array(list(itertools.product(*[self.dims, self.no_of_training_samples, self.no_of_layers])), dtype=np.float32)
        done = np.zeros((len(middle_block), 1), dtype=np.float32)
        metrics = np.zeros((len(middle_block), len(self.metrics)), dtype=np.float32)
        values = np.concatenate([done, middle_block, metrics], axis=1)
        self.df: pd.DataFrame = pd.DataFrame(data=values, columns=['done', 'dim', 'samples', 'layers'] + self.metrics)


class KLExperiment(MafExperiment):
    def __init__(self, name: str, csv_file: Optional[Path] = None):
        super().__init__(name)
        self.plan: Optional[Plan] = self.create_plan()
        self.csv_file: Optional[Path] = csv_file

    def create_data_distribution(self, dim: int) -> Distribution:
        raise NotImplementedError()

    def create_maf(self, dim: int, layers: int) -> MaskedAutoregressiveFlow:
        raise NotImplementedError()

    def create_classifier(self) -> LazyModel:
        raise NotImplementedError()

    def create_plan(self) -> Plan:
        raise NotImplementedError()

    def _run(self):
        self.plan = self.create_plan()
        for index, row in self.plan.df.iterrows():
            dim: int = int(row['dim'])
            samples: int = int(row['samples'])
            layers: int = int(row['layers'])
            distribution: Distribution = self.create_data_distribution(dim=dim)
            maf = self.create_maf(dim=dim, layers=layers)
            maf.fit(dataset=distribution.sample(samples), epochs=30, batch_size=10)
            kld = KullbackLeiblerDivergence(p=distribution, q=maf, half_width=self.plan.kl_half_width, step_size=self.plan.kl_step_size, batch_size=self.plan.kl_batch_size)
            kl = kld.calculate()
            # kl_tf = maf.transformed_distribution.kl_divergence(distribution.tfd_distribution)
            # kl_tf = distribution.tfd_distribution.kl_divergence(maf.transformed_distribution)
            # print(f"{samples} samples -> {kl_tf} kl_tf")
            print(f"{samples} samples -> {kl} kl")
            self.plan.df.at[index, 'kl_divergence'] = kl
        if self.csv_file is not None:
            self.plan.df.to_csv(self.csv_file)
        print(self.plan.df)


class KLTest(KLExperiment):
    def __init__(self):
        super().__init__('just_a_test', csv_file=Path('test_csv.csv'))

    def create_plan(self) -> Plan:
        return Plan(dims=[2], no_of_training_samples=[100 * i for i in range(1, 11)], no_of_layers=[2], kl_half_width=7.0, kl_step_size=.1)

    def create_data_distribution(self, dim: int) -> Distribution:
        return MultimodalDistribution(input_dim=dim, distributions=[GaussianMultivariate(input_dim=dim, mus=[-2.5] * dim, cov=[1] * dim),
                                                                    GaussianMultivariate(input_dim=dim, mus=[2.5] * dim, cov=[1] * dim)])
        # return GaussianMultivariate(input_dim=dim, mus=[-2.5] * dim, cov=[1] * dim)

    def create_maf(self, dim: int, layers: int) -> MaskedAutoregressiveFlow:
        return MaskedAutoregressiveFlow(input_dim=dim, layers=layers, hidden_shape=[200, 200], norm_layer=True)


if __name__ == '__main__':
    enable_memory_growth()
    s = KLTest()
    s.run()
