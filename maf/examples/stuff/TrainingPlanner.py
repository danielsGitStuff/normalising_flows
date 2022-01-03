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


class TrainingPlanner:
    def __init__(self, dataset_size_start: int,
                 dataset_size_end: int,
                 dataset_size_steps: int = 10,
                 synth_ratio_start: float = 0.0,
                 synth_ratio_end: float = 1.0,
                 synth_ratio_steps: int = 10,
                 classifiers_per_nf: int = 5,
                 val_size: Optional[int] = None,
                 val_split: Optional[float] = .1):
        self.val_size: Optional[int] = val_size
        self.val_split: Optional[float] = val_split
        if val_split is not None and val_size is not None:
            raise RuntimeError("do not set 'val_size' and 'val_split' at once")
        self.training_size_start: int = dataset_size_start
        self.training_size_end: int = dataset_size_end
        self.training_size_steps: int = dataset_size_steps
        self.synth_ratio_start: float = synth_ratio_start
        self.synth_ratio_end: float = synth_ratio_end
        self.synth_ratio_steps: int = synth_ratio_steps
        self.classifiers_per_nf: int = classifiers_per_nf
        self.metrics: List[str] = ['accuracy', 'loss', 'max_epoch', 'tnoise', 'fnoise', 'tsig', 'fsig']
        self.dataset_sizes: np.ndarray = np.ceil(np.linspace(self.training_size_start, self.training_size_end, self.training_size_steps), dtype=np.float32)
        self.synth_ratios: np.ndarray = np.linspace(self.synth_ratio_start, self.synth_ratio_end, self.synth_ratio_steps, dtype=np.float32)
        self.classifier_numbers: np.ndarray = np.linspace(0, self.classifiers_per_nf - 1, self.classifiers_per_nf, dtype=np.float32)
        middle_block: np.ndarray = np.array(list(itertools.product(*[self.dataset_sizes, self.synth_ratios, self.classifier_numbers])))

        d_t_map: Dict[int, int] = {}
        d_v_map: Dict[int, int] = {}
        val_size = self.val_size
        for d_size in self.dataset_sizes:
            if self.val_split is not None:
                val_size = math.floor(d_size * self.val_split)
                train_size = d_size - val_size
            else:
                train_size = d_size - val_size
            if train_size <= 0:
                raise ValueError("training size <= 0. Increase dataset size or reduce val_size")
            d_t_map[d_size] = train_size
            d_v_map[d_size] = val_size
        self.val_sizes: List[int] = [d_v_map[d] for d in self.dataset_sizes]
        self.train_sizes: List[int] = [d_t_map[d] for d in self.dataset_sizes]

        val_sizes: List[int] = [d_v_map[d] for d in middle_block[:, 0]]
        train_sizes: List[int] = [d_t_map[d] for d in middle_block[:, 0]]

        train_val: np.ndarray = np.column_stack([train_sizes, val_sizes]).astype(np.float32)
        done = np.zeros((len(middle_block), 1), dtype=np.float32)
        metrics = np.zeros((len(middle_block), len(self.metrics)), dtype=np.float32)
        values = np.concatenate([done, train_val, middle_block, metrics], axis=1)
        self.plan: pd.DataFrame = pd.DataFrame(values, columns=['done', 'tsize', 'vsize', 'dsize', 'synthratio', 'model'] + self.metrics) \
            .sort_values(by=['dsize', 'synthratio', 'model'], ascending=False)
        print(self.plan.head(10))
        print(
            f"training plan requires training {len(self.dataset_sizes)} Normalising Flows "
            f"and training {len(self.dataset_sizes) * len(self.synth_ratios) * self.classifiers_per_nf} classifiers.")

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


if __name__ == '__main__':
    TrainingPlanner(1000, 1200)
