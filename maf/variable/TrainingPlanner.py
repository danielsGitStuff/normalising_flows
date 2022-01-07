from __future__ import annotations
import itertools
import math
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Generator, Set

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series

from maf.examples.stuff.StaticMethods import StaticMethods
from maf.variable.VariableParam import LambdaParam, LambdaParams, VariableParamInt, MetricParam, CopyFromParam, VariableParam, FixedParam
import matplotlib.ticker as ticker


class TrainingPlanner:
    def __init__(self,
                 *params,
                 classifiers_per_nf: int = 5):
        self.classifiers_per_nf: int = classifiers_per_nf
        self._var_params: List[VariableParam] = [p for p in params if isinstance(p, VariableParam)]
        self._fixed_params: List[FixedParam] = [p for p in params if isinstance(p, FixedParam)]
        self._metric_params: List[MetricParam] = [p for p in params if isinstance(p, MetricParam)]
        self._lambda_params: List[LambdaParam] = [p for p in params if isinstance(p, LambdaParam)]
        self.plan: Optional[pd.DataFrame] = None
        self.metrics: List[str] = [p.name for p in self._metric_params]
        self.label_map: Dict[str, str] = {}

    def build_plan(self) -> TrainingPlanner:
        var_params: List[List[float]] = [list(p.get_range()) for p in self._var_params]
        variable_block = np.array(list(itertools.product(*(var_params))))
        fixed_block: np.ndarray = np.array([[p.value] * len(variable_block) for p in self._fixed_params], dtype=np.float32).T
        metric_block: np.ndarray = np.array([[p.value] * len(variable_block) for p in self._metric_params], dtype=np.float32).T
        lambda_block: np.ndarray = np.array([[-2.0] * len(variable_block) for p in self._lambda_params], dtype=np.float32).T
        values = np.hstack([fixed_block, variable_block, lambda_block, metric_block])
        columns: List[str] = [p.name for p in self._fixed_params] + \
                             [p.name for p in self._var_params] + \
                             [p.name for p in self._lambda_params] + \
                             [p.name for p in self._metric_params]
        self.plan = pd.DataFrame(values, columns=columns, dtype=np.float32)
        if len(self._lambda_params) > 0:
            for index, row in self.plan.iterrows():
                for lb in self._lambda_params:
                    self.plan.at[index, lb.name] = lb.f(row[lb.source_params])
        return self

    def print(self, target_file: Path):
        df: pd.DataFrame = self.plan
        group_by: List[str] = [p.name for p in self._var_params if p.name != 'model']
        print(f"print group_by: {group_by}")
        means: pd.DataFrame = df.drop(['tsize', 'vsize'], axis=1).groupby(group_by).mean()
        stddevs: pd.DataFrame = df.drop(['tsize', 'vsize'], axis=1).groupby(group_by).std()
        group_by_operations: List[Tuple[str, pd.DataFrame]] = [('Mean', means), ('Std', stddevs)]
        # metrics = list(df.columns)[6:]
        models_per_config = len(df['model'].unique())
        # {(GROUP_BY_OP, METRIC), pd.DataFrame}
        d: Dict[Tuple[str, str], pd.DataFrame] = {}
        for group_by_operation_name, group_by_df in group_by_operations:
            for metric in self.metrics:
                multi_index: pd.MultiIndex = group_by_df.axes[0]
                group_by_values = np.array([[x, y] for x, y in multi_index], dtype=np.float32)
                metric_values: np.ndarray = group_by_df[metric].values
                values = np.concatenate([group_by_values, metric_values.reshape(len(metric_values), 1)], axis=1)
                processed = pd.DataFrame(data=values, columns=group_by + ['metric'])
                d[(group_by_operation_name, metric)] = processed
        plt.clf()
        big = 24
        medium = 20
        small = 20
        plt.rc('font', size=small)
        plt.rc('axes', titlesize=small)
        plt.rc('axes', labelsize=medium)
        plt.rc('xtick', labelsize=small)
        plt.rc('ytick', labelsize=small)
        plt.rc('legend', fontsize=small)
        plt.rc('figure', titlesize=big)
        plt.rc('lines', linewidth=3)
        fig, axs = StaticMethods.default_fig(no_rows=len(group_by_operations), no_columns=len(self.metrics), w=14, h=11)
        fig.suptitle(f"Results for {models_per_config} classifiers")
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for ax in axs.flatten():
            ax.set_axis_off()
        for ((group_by_operation_name, metric), df), ax in zip(d.items(), axs.flatten()):
            ax.set_axis_on()
            if 'dsize' in df:
                df['dsize'] = df['dsize'].astype(np.int32)
            # fig, ax = StaticMethods.default_fig(no_rows=1, no_columns=1, w=10, h=8)
            ax.set_title(f"{group_by_operation_name} of '{metric}'")
            pivoted = df.pivot(index=group_by[0], columns=group_by[1], values='metric').T[::-1]

            sns.heatmap(data=pivoted, annot=True, fmt='.2f', ax=ax, square=True, )
            ax.xaxis.set_major_formatter(ticker.EngFormatter(places=3))
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{}".format(x)))
            ys: List[float] = list(pivoted.index)
            ax.set_yticks(list(range(len(ys))))
            ax.set_yticklabels(["{:.2f}".format(y) for y in ys])

            xs: List[float] = list(pivoted.columns)
            ax.set_xticks(list(range(len(xs))))
            ax.set_xticklabels(["{:.2f}".format(x) for x in xs])
            # ax.set_xlabel(self.label_map.get(group_by[0], group_by[0]))
            # ax.set_ylabel(self.label_map.get(group_by[1], group_by[1]))

            # ax.set_xlabel('Synthetic Samples ratio')
            # ax.set_ylabel('No of Training Samples')
            plt.tight_layout()
        plt.savefig(target_file)

    def print_confusion_matrices(self, target_file: Path):
        df: pd.DataFrame = self.plan
        group_by: List[str] = [p.name for p in self._var_params if p.name != 'model']
        print(f"print_confusion_matrices group_by: {group_by}")
        means: pd.DataFrame = df.groupby(group_by).mean()
        stddevs: pd.DataFrame = df.groupby(group_by).std()
        group_by_operations: List[Tuple[str, pd.DataFrame]] = [('Mean', means), ('Std', stddevs)]
        # metrics = list(df.columns)[6:]
        models_per_config = len(df['model'].unique())
        # {(GROUP_BY_OP, METRIC), pd.DataFrame}
        d: Dict[Tuple[str, str], pd.DataFrame] = {}
        metrics: Set = set(self.metrics)
        for m in ['tsig', 'fsig', 'tnoise', 'fnoise']:
            if m not in metrics:
                print(f"metric '{m}' not present. aborting", file=sys.stderr)
                return
        metrics: List[str] = list(sorted(metrics))
        for group_by_operation_name, group_by_df in group_by_operations:
            for metric in ['tsig', 'fsig', 'tnoise', 'fnoise']:
                multi_index: pd.MultiIndex = group_by_df.axes[0]
                group_by_values = np.array([[x, y] for x, y in multi_index], dtype=np.float32)
                metric_values: np.ndarray = group_by_df[metric].values
                values = np.concatenate([group_by_values, metric_values.reshape(len(metric_values), 1)], axis=1)
                processed = pd.DataFrame(data=values, columns=group_by + ['metric'])
                processed = processed.pivot(index=group_by[0], columns=group_by[1], values='metric').T[::-1]
                d[(group_by_operation_name, metric)] = processed
        plt.clf()
        big = 24
        medium = 20
        small = 20
        plt.rc('font', size=small)
        plt.rc('axes', titlesize=small)
        plt.rc('axes', labelsize=medium)
        plt.rc('xtick', labelsize=small)
        plt.rc('ytick', labelsize=small)
        plt.rc('legend', fontsize=small)
        plt.rc('figure', titlesize=big)
        plt.rc('lines', linewidth=3)
        shape = d[('Mean', 'tsig')].shape
        fig, axs = StaticMethods.default_fig(no_rows=shape[0], no_columns=shape[1], w=10, h=8)
        maxis = [d[('Mean', 'tsig')].max().max(), d[('Mean', 'fsig')].max().max(), d[('Mean', 'tnoise')].max().max(), d[('Mean', 'fnoise')].max().max()]
        v_max = max(maxis)

        for row in range(shape[0]):
            for column in range(shape[1]):
                t_sig = d[('Mean', 'tsig')].values[row][column]
                f_sig = d[('Mean', 'fsig')].values[row][column]
                t_noise = d[('Mean', 'tnoise')].values[row][column]
                f_noise = d[('Mean', 'fnoise')].values[row][column]
                ax = axs[row][column]
                df = pd.DataFrame(np.array([[t_sig, f_noise], [f_sig, t_noise]]), dtype=np.float32)
                sns.heatmap(data=df, ax=ax, annot=True, fmt='.2f', square=True, xticklabels=['T', 'N'], yticklabels=['T', 'N'], vmax=v_max, cbar=False)
                # ax.set_xlabel('Prediction')
                # ax.set_ylabel('Truth')
        models_per_config = len(self.plan['model'].unique())
        fig.suptitle(f"Confusion matrices (average of {models_per_config} classifiers). Vertical: Truth, horizontal: Prediction")
        fig.supxlabel(self.label_map.get(group_by[0], group_by[0]))
        fig.supylabel(self.label_map.get(group_by[1], group_by[1]))
        proxy_df: pd.DataFrame = d[('Mean', 'tsig')]
        for i, ax in enumerate(axs[:, 0]):
            ax.set_ylabel(math.floor(proxy_df.axes[0][i]))
        for i, ax in enumerate(axs[-1, :]):
            ax.set_xlabel(math.floor(proxy_df.axes[1][i]))
        plt.tight_layout()
        plt.savefig(target_file)

    def get_classifier_name(self, row: Series, extension: Optional[str] = None) -> str:
        """@return the base file name for everything that just depends on a classifier (witch itself depends on a NF): training history, model file"""
        appendices: List[str] = [f"{p.name}={row[p.name]}" for p in self._var_params]
        appendix: str = ','.join(appendices)
        name = f"cl_{appendix}"
        if extension is not None:
            name = f"{name}.{extension}"
        return name

    def label(self, param: str, label: str) -> TrainingPlanner:
        self.label_map[param] = label
        return self


if __name__ == '__main__':
    p = TrainingPlanner(FixedParam('done', 0),
                        LambdaParams.tsize_from_dsize(val_size=1500),
                        LambdaParams.vsize_from_dsize(val_size=1500),
                        VariableParam('dsize', range_start=2500, range_end=25000, range_steps=3),
                        VariableParam('synthratio', range_start=0.0, range_end=1.0, range_steps=3),
                        VariableParamInt('model', range_start=0, range_end=3, range_steps=3),
                        MetricParam('loss'),
                        MetricParam('accuracy'),
                        MetricParam('max_epoch'),
                        MetricParam('tnoise'),
                        MetricParam('fnoise'),
                        MetricParam('tsig'),
                        MetricParam('fsig'),
                        CopyFromParam('clfsize', source_param='dsize'))
    p.build_plan()
    print('end')
