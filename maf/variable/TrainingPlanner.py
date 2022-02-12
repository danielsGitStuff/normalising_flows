from __future__ import annotations

import itertools
import sys

import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series

from maf.stuff.StaticMethods import StaticMethods
from maf.variable.DependencyChecker import Dependency, DependencyChecker
from maf.variable.VariableParam import LambdaParam, MetricParam, VariableParam, FixedParam, Param, MetricIntParam, Converter
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set


class TrainingPlanner:
    def __init__(self,
                 *params,
                 classifiers_per_nf: int = 5):
        self.classifiers_per_nf: int = classifiers_per_nf
        params: List[Param] = list(params)
        self._var_params: List[VariableParam] = [p for p in params if isinstance(p, VariableParam)]
        self._fixed_params: List[FixedParam] = [p for p in params if isinstance(p, FixedParam)]
        self._metric_params: List[MetricParam] = [p for p in params if isinstance(p, (MetricParam, MetricIntParam))]
        self._lambda_params: List[LambdaParam] = [p for p in params if isinstance(p, LambdaParam)]
        self._varying_params: List[Param] = [p for p in params if p.is_var]
        self._param_dict: Dict[str, Param] = {p.name: p for p in params}
        self.plan: Optional[pd.DataFrame] = None
        self.metrics: List[str] = [p.name for p in self._metric_params]
        self.label_map: Dict[str, str] = {}
        self.built: bool = False
        self.sanity_check()

    def sanity_check(self):
        names: Set[str] = set()
        dependencies_out: Dict[str, Set[str]] = {}
        dependencies_in: Dict[str, Set[str]] = {}

        def put(params: List[Param]):
            for p in params:
                if p.name in names:
                    raise RuntimeError(f"param '{p.name}' has been defined twice. at least.")
                names.add(p.name)

        put(self._var_params)
        put(self._fixed_params)
        put(self._metric_params)
        put(self._lambda_params)

        dependencies: List[Dependency] = [p.to_dependency() for p in self._lambda_params + self._var_params + self._fixed_params]
        checker = DependencyChecker()
        checker.check_dependencies(dependencies)

    def build_plan(self) -> TrainingPlanner:
        if self.built:
            return self
        self.built = True
        var_params: List[List[float]] = [list(p.get_range()) for p in self._var_params]
        variable_block = np.array(list(itertools.product(*(var_params))))
        fixed_block: np.ndarray = np.array([[p.value] * len(variable_block) for p in self._fixed_params], dtype=np.float32).T
        metric_block: np.ndarray = np.array([[-1.0] * len(variable_block) for p in self._metric_params], dtype=np.float32).T
        lambda_block: np.ndarray = np.array([[-2.0] * len(variable_block) for p in self._lambda_params], dtype=np.float32).T
        log_block: np.ndarray = np.ndarray
        # throw out everything that is empty
        blocks = [block for block in [fixed_block, variable_block, lambda_block, metric_block] if len(block) > 0]
        # values = np.hstack([fixed_block, variable_block, lambda_block, metric_block])
        values = np.hstack(blocks)
        columns: List[str] = [p.name for p in self._fixed_params] + \
                             [p.name for p in self._var_params] + \
                             [p.name for p in self._lambda_params] + \
                             [p.name for p in self._metric_params]
        self.plan = pd.DataFrame(values, columns=columns, dtype=np.float32)
        available_columns: Set[str] = set()
        for p in self._var_params + self._fixed_params:
            available_columns.add(p.name)
        remaining_lambdas: Dict[str, LambdaParam] = {p.name: p for p in self._lambda_params}

        def check_req(p: LambdaParam):
            for dependency in p.source_params:
                if dependency not in available_columns:
                    return False
            return True

        while len(remaining_lambdas) > 0:
            for name, param in remaining_lambdas.copy().items():
                if not check_req(param):
                    continue
                for index, row in self.plan.iterrows():
                    series: Series = row[param.source_params]
                    # ps: Dict[str, Any] = {p: series[p] for p in param.source_params}
                    tup = tuple([series[p] for p in param.source_params])
                    self.plan.at[index, name] = param.f(*tup)
                    # self.plan.at[index, name] = param.f(row[param.source_params])
                available_columns.add(name)
                del remaining_lambdas[name]
        self.plan = self.plan.reindex(sorted(self.plan.columns), axis=1)
        self.plan = self.plan[['done', 'model'] + [col for col in self.plan.columns if col not in ['done', 'model']]]
        metrics = set([p.name for p in self._metric_params])
        self.plan = self.plan[[col for col in self.plan.columns if col not in metrics] + sorted(metrics)]
        # if len(self._lambda_params) > 0:
        #     for index, row in self.plan.iterrows():
        #         for lb in self._lambda_params:
        #             self.plan.at[index, lb.name] = lb.f(row[lb.source_params])
        return self

    def print(self, target_file: Path):
        df: pd.DataFrame = self.plan.copy()
        type_dict: Dict[str, Converter] = {k: v.converter for k, v in self._param_dict.items() if v.converter is not None}
        group_by: List[str] = [p.name for p in self._varying_params if p.name != 'model']
        if len(group_by) < 2:
            group_by.append('dsize')
        print(f"print group_by: {group_by}")
        means: pd.DataFrame = df.drop(['tsize'], axis=1).groupby(group_by).mean()
        stddevs: pd.DataFrame = df.drop(['tsize'], axis=1).groupby(group_by).std()
        group_by_operations: List[Tuple[str, pd.DataFrame]] = [('Mean', means), ('Std', stddevs)]
        models_per_config = len(df['model'].unique())
        d: Dict[Tuple[str, str], pd.DataFrame] = {}
        for group_by_operation_name, group_by_df in group_by_operations:
            for metric in self.metrics:
                multi_index: pd.MultiIndex = group_by_df.axes[0]
                group_by_values = np.array([[x, y] for x, y in multi_index], dtype=np.float32)
                metric_values: np.ndarray = group_by_df[metric].values
                values = np.concatenate([group_by_values, metric_values.reshape(len(metric_values), 1)], axis=1)
                processed = pd.DataFrame(data=values, columns=group_by + ['metric'])
                current_type_dict: Dict[str, Converter] = {g: type_dict[g] for g in group_by if g in type_dict}
                processed = processed.astype(current_type_dict)
                d[(group_by_operation_name, metric)] = processed
        plt.clf()
        L = 20
        M = 16
        S = 12
        plt.rc('font', size=M)
        plt.rc('axes', titlesize=L)
        plt.rc('axes', labelsize=L)
        plt.rc('xtick', labelsize=S)
        plt.rc('ytick', labelsize=S)
        plt.rc('legend', fontsize=S)
        plt.rc('figure', titlesize=L)
        plt.rc('lines', linewidth=3)

        fig, axs = StaticMethods.default_fig(no_rows=len(group_by_operations), no_columns=len(self.metrics), w=13, h=11)
        fig.suptitle(f"Results for {models_per_config} classifiers")
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for ax in axs.flatten():
            ax.set_axis_off()
        for ((group_by_operation_name, metric), df), ax in zip(d.items(), axs.flatten()):
            # if metric != 'accuracy':
            #     continue
            ax.set_axis_on()
            if 'dsize' in df:
                df['dsize'] = df['dsize'].astype(np.int32)
            ax.set_title(f"{group_by_operation_name} of '{metric}'")
            pivoted = df.pivot(index=group_by[0], columns=group_by[1], values='metric').T[::-1]
            if 0.0 in pivoted.index and 0.0 in pivoted[pivoted.columns[0]]:  # and group_by_operation_name == 'Mean':
                pivoted.at[0, 0] = None
            fmt = '.3f'
            mv = np.nanmax(pivoted.values)
            if 100 <= mv <= 1000:
                fmt = '.2f'
            elif mv > 1000:
                fmt = '.0f'
            sns.heatmap(data=pivoted, annot=True, fmt=fmt, ax=ax, square=True, )
            if group_by[0] in self.label_map:
                ax.set_xlabel(self.label_map[group_by[0]])
            if group_by[1] in self.label_map:
                ax.set_ylabel(self.label_map[group_by[1]])
            plt.tight_layout()
        plt.savefig(target_file)

    def print_confusion_matrices(self, target_file: Path):
        df: pd.DataFrame = self.plan
        group_by: List[str] = [p.name for p in self._varying_params if p.name != 'model']
        if len(group_by) < 2:
            group_by.append('dsize')
        print(f"print_confusion_matrices group_by: {group_by}")
        means: pd.DataFrame = df.groupby(group_by).mean()
        stddevs: pd.DataFrame = df.groupby(group_by).std()
        group_by_operations: List[Tuple[str, pd.DataFrame]] = [('Mean', means), ('Std', stddevs)]
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
        axs = axs.reshape(shape)
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
                sns.heatmap(data=df, ax=ax, annot=True, fmt='.2f', square=True, xticklabels=['Signal', 'Noise'], yticklabels=['Signal', 'Noise'], vmax=v_max, cbar=False)
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
