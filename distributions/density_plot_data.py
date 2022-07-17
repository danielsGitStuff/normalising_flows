from __future__ import annotations

import math
import pathlib
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Colormap, TwoSlopeNorm
from plotly import graph_objects as go

from common import jsonloader
from common.NotProvided import NotProvided


class DensityPlotData:
    class Methods:
        @staticmethod
        def load(f: Union[str, Path]) -> DensityPlotData:
            if isinstance(f, str):
                f = pathlib.Path(f)
            plot_data: DensityPlotData = jsonloader.load_json(f)
            return plot_data

        @staticmethod
        def save(plot_data: DensityPlotData, f: Union[str, Path]):
            if isinstance(f, str):
                f = pathlib.Path(f)
            jsonloader.to_json(plot_data, file=f)

    def __init__(self, values: np.ndarray, mesh_count: int, xmin: float = -4.0, xmax: float = 4.0, ymin: Optional[float] = -4.0, ymax: Optional[float] = 4.0,
                 suptitle: Optional[str] = None,
                 title: Optional[str] = None, type: str = "hm", columns: Optional[List[str]] = NotProvided(), truth: Optional[np.ndarray] = None):
        self.mesh_count: int = mesh_count
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.ymin: float = ymin
        self.ymax: float = ymax
        self.values: np.ndarray = values
        self.truth: Optional[np.ndarray] = truth
        self.suptitle: Optional[str] = suptitle
        self.title: Optional[str] = title
        self.type: str = type
        self.columns: Optional[List[str]] = columns
        self.vmin: float = 0.0
        if type not in {"hm", "scatter", "1d", 'diff'}:
            raise RuntimeError(f"unknown plot type: {type}")

    def save(self, f: Union[str, Path]) -> DensityPlotData:
        DensityPlotData.Methods.save(self, f)
        return self

    def print_yourself_3d(self, title: str, show: bool = False, image_base_path: Optional[str] = None):
        x, y = np.linspace(self.xmin, self.xmax, self.mesh_count), np.linspace(self.ymax, self.ymin, self.mesh_count)
        p = np.clip(self.values, -10.0, math.inf)

        if self.truth is not None:
            truth = np.clip(self.truth, -10, math.inf)
            fig = go.Figure(data=[go.Surface(z=p, x=x, y=y), go.Surface(z=truth, x=x, y=y, colorscale='Viridis', showscale=False, opacity=0.5)])
        else:
            fig = go.Figure(data=[go.Surface(z=p, x=x, y=y)])

        fig.update_layout(title=title, autosize=True, margin=dict(l=65, r=50, b=65, t=90))
        if image_base_path is not None:
            fig.write_image(f"{image_base_path}.3d.png", width=1600, height=1600)
            fig.write_html(f"{image_base_path}.3d.html")
        if show:
            fig.show()

    def print_yourself(self, ax, vmax: Optional[float] = None, vmin: Optional[float] = None, cmap: Optional[Colormap] = None, legend: bool = NotProvided,
                       norm: Optional[TwoSlopeNorm] = None):

        if self.title is not None:
            ax.set_title(self.title)
        ax.set_xlim([self.xmin, self.xmax])
        ax.set_ylim([self.ymin, self.ymax])
        ymin = self.ymin if self.ymin is not None else self.values[:, -1].min()
        ymax = self.ymax if self.ymax is not None else self.values[:, -1].max()
        x_tick_labels = [self.xmin, self.xmin / 2, 0.0, self.xmax / 2, self.xmax]
        y_tick_labels = [ymax, ymax / 2, 0, ymin / 2, ymin]
        x_tick_indices = [0] + [int(self.mesh_count * m) for m in [1 / 4, 1 / 2, 3 / 4, 1]]
        y_tick_indices = x_tick_indices
        cbar_kws = {"shrink": .7}
        if self.type == "hm":

            ax.set_aspect("equal")
            # TODO make this work from ranges like [0, 5] not just [-5, 5]
            sns.heatmap(self.values, ax=ax, vmax=vmax, cmap=cmap, vmin=vmin, cbar_kws=cbar_kws, norm=norm)
            ax.set_xticks(x_tick_indices)
            ax.set_xticklabels(x_tick_labels)
            ax.set_yticks(y_tick_indices)
            ax.set_yticklabels(y_tick_labels)

        elif self.type == "scatter":
            df: pd.DataFrame = pd.DataFrame(columns=["x", "y"])
            df["x"] = self.values[:, 0]
            df["y"] = self.values[:, 1]
            sns.scatterplot(data=df, x="x", y="y", ax=ax)
        elif self.type == '1d':
            if vmax is not None:
                ymax = vmax
            if vmin is not None:
                ymin = vmin
            xs = self.values[:, 0]
            ys = self.values[:, 1]
            columns: List[str] = NotProvided.value_if_not_provided(self.columns, ['x', 'y'])
            # ax.plot(xs, ys)
            df = pd.DataFrame(self.values, columns=columns).set_index(columns[0])
            ax.set_ylim(ymin, ymax)
            sns.lineplot(data=df, ax=ax, legend=NotProvided.value_if_not_provided(legend, False))
        elif self.type == 'diff':
            ax.set_aspect("equal")
            # TODO make this work from ranges like [0, 5] not just [-5, 5]
            cbar = NotProvided.value_if_not_provided(legend, True)
            sns.heatmap(self.values, ax=ax, vmax=vmax, cmap=cmap, vmin=vmin, cbar=cbar, cbar_kws=cbar_kws, norm=norm)
            ax.set_xticks(x_tick_indices)
            ax.set_xticklabels(x_tick_labels)
            ax.set_yticks(y_tick_indices)
            ax.set_yticklabels(y_tick_labels)

        if NotProvided.is_provided(self.columns):
            ax.set_xlabel(self.columns[0])
            ax.set_ylabel(self.columns[1])


