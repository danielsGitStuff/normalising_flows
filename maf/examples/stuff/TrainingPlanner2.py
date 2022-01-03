from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from maf.examples.stuff.StaticMethods import StaticMethods
from maf.mixlearn.VariationalParam import VariationalParam


class VariableBox:
    def __init__(self, params: List[VariationalParam]):
        self.params: List[VariationalParam] = params
        self.__m: Dict[str, VariationalParam] = {p.name: p for p in params}
        self.__indices: Dict[str, int] = {p.name: i for i, p in enumerate(params)}
        self.__indices['model'] = len(self.__indices)

    def get_index(self, name: str) -> int:
        return self.__indices[name]

    def get_range(self, name: str) -> np.ndarray:
        p: VariationalParam = self.__m[name]
        return p.get_range()

    def get_variable_block(self, clf_per_combination: int) -> np.ndarray:
        classifier_numbers = list(np.linspace(0, clf_per_combination - 1, clf_per_combination, dtype=np.float32))
        ranges: List[List[float]] = [list(p.get_range()) for p in self.params]
        middle_block: np.ndarray = np.array(list(itertools.product(*(ranges + [classifier_numbers]))))
        return middle_block

    def get_variable_columns(self) -> List[str]:
        columns: List[str] = [p.name for p in self.params] + ['model']
        return columns


class TrainingPlanner2:
    def __init__(self,
                 params: List[VariationalParam],
                 classifiers_per_nf: int = 5,
                 val_size: Optional[int] = None,
                 val_split: Optional[float] = .1):
        """@param variational_parameters: Provide a List of TWO variational parameters here. If you want to vary the data set size, create a VariationalParam with name='dsize'.
        Otherwise set fixed_fixed_dataset_size """
        self.box: VariableBox = VariableBox(params)
        box = self.box
        self.val_size: Optional[int] = val_size
        self.val_split: Optional[float] = val_split
        if val_split is not None and val_size is not None:
            raise RuntimeError("do not set 'val_size' and 'val_split' at once")
        self.classifiers_per_nf: int = classifiers_per_nf
        self.metrics: List[str] = ['accuracy', 'loss', 'max_epoch', 'tnoise', 'fnoise', 'tsig', 'fsig']

        vds: Optional[VariationalParam] = None
        for vp in params:
            if vp.name == 'dsize':
                vds = vp
                break
        dataset_sizes = box.get_range('dsize')
        self.ranges: List[np.ndarray] = [dataset_sizes]

        # self.classifier_numbers: np.ndarray = np.linspace(0, self.classifiers_per_nf - 1, self.classifiers_per_nf, dtype=np.float32)
        # middle_block: np.ndarray = np.array(list(itertools.product(*[self.dataset_sizes, self.synth_ratios, self.classifier_numbers])))
        # middle_block: np.ndarray = np.array(list(itertools.product(*(self.ranges + [self.classifier_numbers]))))
        middle_block = box.get_variable_block(clf_per_combination=self.classifiers_per_nf)

        dsize_index = box.get_index('dsize')

        # def calculate_train_val_sizes(ds_sizes: np.ndarray) -> Tuple[List[int], List[int]]:
        #     d_t_map: Dict[int, int] = {}
        #     d_v_map: Dict[int, int] = {}
        #     val_size = self.val_size
        #     for d_size in ds_sizes:
        #         if self.val_split is not None:
        #             val_size = math.floor(d_size * self.val_split)
        #             train_size = d_size - val_size
        #         else:
        #             train_size = d_size - val_size
        #         # if train_size <= 0:
        #         #     raise ValueError("training size <= 0. Increase dataset size or reduce val_size")
        #         d_t_map[d_size] = train_size
        #         d_v_map[d_size] = val_size
        #     val_sizes: List[int] = [d_v_map[d] for d in ds_sizes]
        #     train_sizes: List[int] = [d_t_map[d] for d in ds_sizes]
        #     return train_sizes, val_sizes
        #
        # train_sizes, val_sizes = calculate_train_val_sizes(dataset_sizes)
        # clf_train_sizes, clf_val_sizes = calculate_train_val_sizes(box.get_range('clfsize'))

        train_val = np.zeros((len(middle_block), 4), dtype=np.float32)
        done = np.zeros((len(middle_block), 1), dtype=np.float32)
        metrics = np.zeros((len(middle_block), len(self.metrics)), dtype=np.float32)
        values = np.concatenate([done, train_val, middle_block, metrics], axis=1)
        columns = ['done', 'tsize', 'vsize', 'clf_t_g_size', 'clf_v_g_size'] + box.get_variable_columns() + self.metrics
        self.plan: pd.DataFrame = pd.DataFrame(values, columns=columns) \
            .sort_values(by=['dsize', 'synthratio', 'model'], ascending=False)

        def calculate_train_val_sizes(ds_sizes: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
            d_t_map: Dict[int, int] = {}
            d_v_map: Dict[int, int] = {}
            val_size = self.val_size
            for d_size in ds_sizes:
                if self.val_split is not None:
                    val_size = math.floor(d_size * self.val_split)
                    train_size = d_size - val_size
                else:
                    train_size = d_size - val_size
                # if train_size <= 0:
                #     raise ValueError("training size <= 0. Increase dataset size or reduce val_size")
                d_t_map[d_size] = train_size
                d_v_map[d_size] = val_size
            return d_t_map, d_v_map

        d_t_map, d_v_map = calculate_train_val_sizes(box.get_range('dsize'))
        v_t_map, v_v_map = calculate_train_val_sizes(box.get_range('clfsize'))

        for index, row in self.plan.iterrows():
            print(row)
            self.plan.at[index, 'tsize'] = d_t_map[row['dsize']]
            self.plan.at[index, 'vsize'] = d_v_map[row['dsize']]

            self.plan.at[index, 'clf_t_g_size'] = v_t_map[row['clfsize']]

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(self.plan.head(10))
        runs_required = self.classifiers_per_nf
        for rang in self.ranges:
            runs_required *= len(rang)
        print(
            f"training plan requires training {len(dataset_sizes)} Normalising Flows "
            f"and training {runs_required} classifiers.")

    def print(self, target_file: Path):
        df: pd.DataFrame = self.plan

        means: pd.DataFrame = df.drop(['tsize', 'vsize'], axis=1).groupby(['dsize', 'synthratio']).mean()
        stddevs: pd.DataFrame = df.drop(['tsize', 'vsize'], axis=1).groupby(['dsize', 'synthratio']).std()
        group_by_operations: List[Tuple[str, pd.DataFrame]] = [('Mean', means), ('Std', stddevs)]
        metrics = list(df.columns)[6:]
        models_per_config = len(df['model'].unique())
        # {(GROUP_BY_OP, METRIC), pd.DataFrame}
        d: Dict[Tuple[str, str], pd.DataFrame] = {}
        for group_by_operation_name, group_by_df in group_by_operations:
            for metric in metrics:
                multi_index: pd.MultiIndex = group_by_df.axes[0]
                tsize_and_synth = np.array([[x, y] for x, y in multi_index], dtype=np.float32)
                metric_values: np.ndarray = group_by_df[metric].values
                values = np.concatenate([tsize_and_synth, metric_values.reshape(len(metric_values), 1)], axis=1)
                processed = pd.DataFrame(data=values, columns=['dsize', 'synthratio', 'metric'])
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
        fig, axs = StaticMethods.default_fig(no_rows=len(group_by_operations), no_columns=len(metrics), w=10, h=8)
        fig.suptitle(f"Results for {models_per_config} classifiers")
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for ax in axs.flatten():
            ax.set_axis_off()
        for ((group_by_operation_name, metric), df), ax in zip(d.items(), axs.flatten()):
            ax.set_axis_on()
            df['dsize'] = df['dsize'].astype(np.int32)
            # fig, ax = StaticMethods.default_fig(no_rows=1, no_columns=1, w=10, h=8)
            ax.set_title(f"{group_by_operation_name} of '{metric}'")
            sns.heatmap(data=df.pivot('synthratio', 'dsize', 'metric').T[::-1], annot=True, fmt='.2f', ax=ax, square=True, )
            ax.set_xlabel('Synthetic Samples ratio')
            ax.set_ylabel('No of Training Samples')
            plt.tight_layout()
        plt.savefig(target_file)

    def get_todo(self) -> pd.DataFrame:
        todo = self.plan.loc[self.plan['done'] < 1.0]
        return todo

    def set_columns_equal(self, source_column: str, target_column: str) -> TrainingPlanner2:
        for index, row in self.plan.iterrows():
            src = row[source_column]
            self.plan.at[index, target_column] = src
        return self

    def iterate(self, *columns):
        print(columns)


if __name__ == '__main__':
    t1 = TrainingPlanner2(params=[VariationalParam(name='dsize', range_start=2500, range_end=25000, range_steps=3),
                                  VariationalParam(name='synthratio', range_start=0, range_end=1, range_steps=3, int_type=False),
                                  VariationalParam.fixed(name='clfsize', value=2000)], classifiers_per_nf=3)
    t2 = TrainingPlanner2(params=[VariationalParam(name='dsize', range_start=2500, range_end=25000, range_steps=3),
                                  VariationalParam(name='synthratio', range_start=0, range_end=1, range_steps=3, int_type=False),
                                  VariationalParam.fixed(name='clfsize', value=-1.0, int_type=False)], classifiers_per_nf=3) \
        .set_columns_equal('dsize', 'clfsize') \
        .set_columns_equal('tsize', 'clf_t_g_size') \
        .set_columns_equal('vsize', 'clf_v_g_size')
    print('end')
