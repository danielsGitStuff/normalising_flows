from __future__ import annotations

import math
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import tensorflow as tf
from tensorflow.python.data import Dataset

from common.util import Runtime
from common.NotProvided import NotProvided
from distributions.LearnedDistribution import LearnedDistribution, EarlyStop
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.Distribution import Distribution
from distributions.GaussianMultiModal import GaussianMultimodal
from distributions.GaussianMultivariate import GaussianMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow, NanError
from maf.pool import RestartingPoolReplacement


class Defaults:
    @staticmethod
    def create_default_model_definitions() -> List[Dict[str, Any]]:
        model_definitions = [{"model": "NF", "parameters": {"batch_size": 50, "layers": 1, "epochs": 200, "hidden_shape": [200, 200]}},
                             {"model": "NF", "parameters": {"batch_size": 50, "layers": 2, "epochs": 200, "hidden_shape": [200, 200]}},
                             {"model": "NF", "parameters": {"batch_size": 50, "layers": 3, "epochs": 200, "hidden_shape": [200, 200]}},
                             {"model": "NF", "parameters": {"batch_size": 50, "layers": 4, "epochs": 200, "hidden_shape": [200, 200]}}]
        return model_definitions

    @staticmethod
    def create_gauss_5_in_a_row_y() -> Distribution:
        source_distribution = GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-10, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[-5, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[5, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[10, 0], cov=[0.3, 0.3])])
        return source_distribution

    @staticmethod
    def create_gauss_5_in_a_row_x() -> Distribution:
        source_distribution = GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[0, -10], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, -5], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 5], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 10], cov=[0.3, 0.3])])
        return source_distribution

    @staticmethod
    def create_gauss_no() -> GaussianMultimodal:
        cov = [0.3, 0.3]
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[-10, -10], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[-7.5, 7.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[-5, -7.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[0, -2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[2.5, 5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[5, -5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[7.5, 0], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[10, 10], cov=cov)
        ])

    @staticmethod
    def create_gauss_no_4() -> GaussianMultimodal:
        cov = [0.3, 0.3]
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[0, -2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[2.5, 5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[5, -5], cov=cov),
        ])

    @staticmethod
    def create_gauss_no_3() -> GaussianMultimodal:
        cov = [0.3, 0.3]
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[0, -2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[2.5, 5], cov=cov)
        ])

    @staticmethod
    def create_gauss_4_diagonal() -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-7.5, 7.5], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[2.5, -2.5], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[7.5, -7.5], cov=[0.3, 0.3])])

    @staticmethod
    def create_1d_2_gauss():
        return MultimodalDistribution(input_dim=1, distributions=[GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[1.0]),
                                                                  GaussianMultivariate(input_dim=1, mus=[2.5], cov=[1.0])])

    @staticmethod
    def create_gauss_1_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_2_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[2.5, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_3_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[5, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_4_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-7.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[-2.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[2.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[7.5, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_5_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-10, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[-5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[10, offset], cov=[0.3, 0.3])])

    @staticmethod
    def swap_xy(multiple_nd_gaussians: GaussianMultimodal) -> GaussianMultimodal:
        gaussians = [GaussianMultivariate(input_dim=2, mus=[g.mus[1], g.mus[0]], cov=g.cov_matrix) for g in multiple_nd_gaussians.distributions]
        m = GaussianMultimodal(input_dim=2, distributions=gaussians)
        return m

    @staticmethod
    def create_biggus_dickus(offset: int = 0) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[0.0, offset], cov=[3, 3]),
            GaussianMultivariate(input_dim=2, mus=[-3, 3 + offset], cov=[2.0, 2.0]),
            GaussianMultivariate(input_dim=2, mus=[-3, -2 + offset], cov=[3.3, 2.3]),
            GaussianMultivariate(input_dim=2, mus=[4, 0 + offset], cov=[2.3, 3.3])
        ])