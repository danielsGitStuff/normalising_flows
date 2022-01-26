from abc import ABC
from pathlib import Path

import numpy as np
from typing import Optional, List

from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.UniformMultivariate import UniformMultivariate
from maf.DL import DL3, DL2, DataSource
from maf.DS import DS
from maf.examples.stuff.StaticMethods import StaticMethods


class ArtificialDL3(DL3, ABC):
    def __init__(self, dl_folder: Path):
        super().__init__(url='no url', dl_folder=dl_folder)
        self.size: int = 111111
        self.snr: float = 0.5
        self.signal_dir: Path = Path(self.dl_folder, 'signal')
        self.noise_dir: Path = Path(self.dl_folder, 'noise')


class ArtificialIntersection2DDL3(ArtificialDL3):
    def __init__(self):
        super().__init__(dl_folder=Path(StaticMethods.cache_dir(), 'artificial_intersection_2d_1'))
        self.signal_distr: MultimodalDistribution = MultimodalDistribution(input_dim=2, distributions=[
            UniformMultivariate(input_dim=2, lows=[-1, -1], highs=[3, 3]),
            UniformMultivariate(input_dim=2, lows=[-1, 4], highs=[0, 5])
        ])
        self.noise_distr: MultimodalDistribution = MultimodalDistribution(input_dim=2, distributions=[
            UniformMultivariate(input_dim=2, lows=[1, 1], highs=[-3, -3]),
            UniformMultivariate(input_dim=2, lows=[1, -4], highs=[2, -5])
        ])

    def fetch_impl(self):
        if DL2.can_load(self.signal_dir) and DL2.can_load(self.noise_dir):
            return
        no_sig: int = round(self.size * self.snr)
        no_noi: int = self.size - no_sig
        signal = self.signal_distr.sample(size=no_sig)
        noise = self.noise_distr.sample(size=no_noi)
        data = np.concatenate([signal, noise])
        normalised, mean, std = StaticMethods.norm(data)
        normalised_signal = normalised[:len(signal), 1:]
        normalised_noise = normalised[len(signal):, 1:]
        normalised_signal = DS.from_tensor_slices(normalised_signal)
        normalised_noise = DS.from_tensor_slices(normalised_noise)
        dl = DL2(dataset_name=self.dl_folder.name,
                 dir=self.dl_folder,
                 signal_source=DataSource(ds=normalised_signal),
                 noise_source=DataSource(ds=normalised_noise),
                 amount_of_signals=len(signal),
                 amount_of_noise=len(noise))
        dl.create_data()
