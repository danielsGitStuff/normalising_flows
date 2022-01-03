from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List

import numpy as np

from common import jsonloader
from common.NotProvided import NotProvided
from distributions.Distribution import TfpD, DensityPlotData
from distributions.LearnedDistribution import LearnedDistribution, LearnedConfig, EarlyStop, LearnedDistributionCreator
from distributions.base import TTensor, TTensorOpt, TDataOpt, cast_dataset_to_tensor, cast_to_ndarray
from maf.DS import DS
from maf.SaveSettings import SaveSettings
from scipy.stats import multivariate_normal


class LearnedGaussianMultivariateCreatorCov(LearnedDistributionCreator):
    def __init__(self):
        super().__init__()

    def create(self, input_dim: int, conditional_dims: int, conditional_classes: List[str, int] = None) -> LearnedDistribution:
        g = LearnedGaussianMultivariateCov(input_dim=input_dim, conditional_dims=conditional_dims)
        return g

    def load(self, folder: Union[Path, str], prefix: str) -> LearnedDistribution:
        f = Path(folder, f"{prefix}.json")
        g: LearnedGaussianMultivariateCov = jsonloader.load_json(f, raise_on_404=True)
        return g


class LearnedGaussianMultivariateCreatorVar(LearnedDistributionCreator):
    def __init__(self):
        super().__init__()

    def create(self, input_dim: int, conditional_dims: int, conditional_classes: List[str, int] = None) -> LearnedDistribution:
        g = LearnedGaussianMultivariateVar(input_dim=input_dim, conditional_dims=conditional_dims)
        return g

    def load(self, folder: Union[Path, str], prefix: str) -> LearnedDistribution:
        f = Path(folder, f"{prefix}.json")
        g: LearnedGaussianMultivariateVar = jsonloader.load_json(f, raise_on_404=True)
        return g


class LearnedGaussianMultivariateCov(LearnedDistribution):
    @staticmethod
    def load(folder: Union[Path, str], prefix: str) -> LearnedDistribution:
        f = Path(folder, f"{prefix}.json")
        g: LearnedGaussianMultivariateCov = jsonloader.load_json(f, raise_on_404=True)
        return g

    def __init__(self, input_dim: int, conditional_dims: int):
        super().__init__(input_dim, conditional_dims)
        self.mean: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None

    def fit(self, dataset: DS, epochs: int, batch_size: Optional[int] = None, val_xs: TDataOpt = None, val_ys: TDataOpt = None, early_stop: Optional[EarlyStop] = None,
            plot_data_every: Optional[int] = None, lr: float = NotProvided(), shuffle: bool = False) -> Optional[List[DensityPlotData]]:
        xs, _ = cast_dataset_to_tensor(dataset)
        xs: np.ndarray = cast_to_ndarray(xs)
        print("asdasd")
        self.mean = np.mean(xs, axis=0)
        self.cov = np.cov(xs, rowvar=False)

    def save(self, folder: Union[Path, str], prefix: str):
        f = Path(folder, f"{prefix}.json")
        jsonloader.to_json(self, f)

    def get_base_name_part(self, save_settings: SaveSettings) -> str:
        return 'base_name_xx4'

    def get_config(self) -> LearnedConfig:
        return None

    def _create_base_distribution(self) -> Optional[TfpD]:
        return None

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        xs: np.ndarray = cast_to_ndarray(xs)
        r = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True).logpdf(xs)
        return cast_to_ndarray(r)

    def _sample(self, size: int = 1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        s = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True).rvs(size)
        return cast_to_ndarray(s)

    def _likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        xs: np.ndarray = cast_to_ndarray(xs)
        r = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True).pdf(xs)
        return cast_to_ndarray(r)


class LearnedGaussianMultivariateVar(LearnedGaussianMultivariateCov):
    def __init__(self, input_dim: int, conditional_dims: int):
        super().__init__(input_dim, conditional_dims)

    def fit(self, dataset: DS, epochs: int, batch_size: Optional[int] = None, val_xs: TDataOpt = None, val_ys: TDataOpt = None, early_stop: Optional[EarlyStop] = None,
            plot_data_every: Optional[int] = None, lr: float = NotProvided(), shuffle: bool = False) -> Optional[List[DensityPlotData]]:
        xs, _ = cast_dataset_to_tensor(dataset)
        xs: np.ndarray = cast_to_ndarray(xs)
        print("asdasd")
        self.mean = np.mean(xs, axis=0)
        self.cov = np.var(xs, axis=0)
