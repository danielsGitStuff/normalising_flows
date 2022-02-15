import numpy as np
from matplotlib import pyplot as plt
from pandas import Series

import argparse
from aftermath.Constants import Constants
from maf.stuff.StaticMethods import StaticMethods
from typing import Dict, Any, List, Tuple, Set

import re
from pathlib import Path
import pandas as pd

import seaborn as sns


class TableMagic2:
    class Methods:
        @staticmethod
        def get_name_from_tablemagic_csv(csv_f: Path) -> str:
            match_primary_name = re.search('.*(?=\.tablemagic\.csv$)', csv_f.name)
            name = csv_f.name[match_primary_name.start():match_primary_name.end()]
            return name

    def __init__(self, directory: Path):
        self.root: Path = directory
        self.cache: Path = Path(directory, '.cache')
        self.experiment_names: Dict[str, str] = Constants.get_name_dict()

    def run(self):
        for o in self.root.iterdir():
            o: Path = o
            if o.is_dir() and o.name.startswith('results_'):
                to_merge: List[Tuple[str, pd.DataFrame]] = []
                for f in o.iterdir():
                    f: Path = f
                    if f.is_file() and f.name.endswith('.tablemagic.csv'):
                        name = TableMagic2.Methods.get_name_from_tablemagic_csv(f)
                        target_file = Path(f.parent, f"{name}.tablemagic.csv")
                        df = pd.read_csv(f)
                        df: pd.DataFrame = pd.read_csv(f)
                        if 'Unnamed: 0' in df.columns:
                            df = df.drop('Unnamed: 0', axis=1)
                        df = df.astype({'layers': int})
                        to_merge.append((name, df))
                if len(to_merge) > 0:
                    # to_merge = []
                    to_merge = [(n, {n: d for n, d in to_merge}[n]) for n in sorted(n for n, _ in to_merge)]  # sort alphabetically with a horrid runtime
                    self.merge(to_merge, target_file=Path(o, f"{o.name}.csv"))

    def merge(self, dfs: List[Tuple[str, pd.DataFrame]], target_file: Path):
        layers: Set[int] = set()
        for name, df in dfs:
            [layers.add(l) for l in df['layers']]
        values: List[List[Any]] = []
        for name, df in dfs:
            ls = df['layers'].unique()
            for l in ls:
                d: pd.DataFrame = df.loc[df['layers'] == l]['kl']
                row = [self.experiment_names.get(name, name), l, d.mean()]
                # row = [name, l, d.mean()]
                values.append(row)

        result: pd.DataFrame = pd.DataFrame(values, columns=['Name', 'Layers', 'avg_kl'])
        pivoted: pd.DataFrame = result.pivot(index='Name', columns=['Layers'], values='avg_kl')
        pivoted.to_csv(target_file, index=False)

        L = 28
        M = 24
        S = 20
        plt.rc('font', size=M)
        plt.rc('axes', titlesize=L)
        plt.rc('axes', labelsize=M)
        plt.rc('xtick', labelsize=S)
        plt.rc('ytick', labelsize=S)
        plt.rc('legend', fontsize=S)
        plt.rc('figure', titlesize=L)
        plt.rc('lines', linewidth=3)

        h_mult = 2 if len(pivoted) < 4 else 1
        fig, ax = StaticMethods.default_fig(1, 1, w=2 * len(layers), h=h_mult * (len(pivoted)))
        fmt = '.2f'
        sns.heatmap(data=pivoted, annot=True, fmt=fmt, ax=ax, square=False, cbar=False, cmap=sns.color_palette("Blues", as_cmap=True))
        ax.set(xlabel='Layers', ylabel=None)
        plt.tight_layout()
        plt.savefig(Path(target_file.parent, f"{target_file.name}.png"))


if __name__ == '__main__':
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument('--dir', help='which dir to traverse', default='pull/', type=str)
    args: Dict[str, Any] = vars(ap.parse_args())
    t = TableMagic2(Path(args['dir']))
    t.run()
