import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from common.util import Runtime
from distributions.Distribution import Distribution


class StaticMethods:

    @staticmethod
    def cache_dir() -> Path:
        cache_dir = Path('.cache')
        if not cache_dir.exists():
            cache_dir.mkdir()
        return cache_dir

    @staticmethod
    def default_fig(no_rows: int, no_columns: int, w: int = 5, h: int = 4, h_offset: int = 0):
        fig, axs = plt.subplots(no_rows, no_columns)
        fig.set_figheight(no_rows * h + h_offset)
        fig.set_figwidth(no_columns * w)
        return fig, axs

    @staticmethod
    def create_grid(from_coord: List[float], to_coord: List[float], step_size: float) -> np.ndarray:
        nr_z_0 = int((to_coord[0] - from_coord[0]) * (1 / step_size))
        nr_z_1 = int((to_coord[1] - from_coord[1]) * (1 / step_size))
        z = np.zeros((nr_z_0, nr_z_1))
        sh_0, sh_1 = z.shape
        x, y = np.linspace(from_coord[0], to_coord[0], sh_0), np.linspace(from_coord[1], to_coord[1], sh_1)
        X, Y = np.meshgrid(x, y)
        xx = X
        yy = Y
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack([r1, r2])

        return grid

    @staticmethod
    def create_data_3d_plot_fast(from_coord: List[float], to_coord: List[float], step_size: float, source_distribution: Optional[Distribution] = None,
                                 learned_distribution: Optional[Distribution] = None) -> Tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        print("sampling for 3d data")
        r = Runtime("sampling").start()
        nr_z_0 = int((to_coord[0] - from_coord[0]) * (1 / step_size))
        nr_z_1 = int((to_coord[1] - from_coord[1]) * (1 / step_size))

        z = np.zeros((nr_z_0, nr_z_1))

        sh_0, sh_1 = z.shape
        x, y = np.linspace(from_coord[0], to_coord[0], sh_0), np.linspace(from_coord[1], to_coord[1], sh_1)
        X, Y = np.meshgrid(x, y)

        xx = X
        yy = Y
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack([r1, r2])
        if source_distribution is None:
            return x, y, None, None

        p_truth = source_distribution.prob(grid)
        p_truth = p_truth.reshape(X.shape)
        p_truth = np.clip(p_truth, -10, math.inf)
        p = learned_distribution.prob(grid, batch_size=10000)
        p = p.reshape(X.shape)
        p = np.clip(p, -10, math.inf)
        r.stop().print()
        print("sampling for 3d data done")
        return x, y, p, p_truth
