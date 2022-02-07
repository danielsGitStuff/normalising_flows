import sys
from typing import Dict, Any, List, Optional

import argparse
import numpy as np
from matplotlib import pyplot as plt

from common.poolreplacement import RestartingPoolReplacement
from keta.argparseer import ArgParser
from maf.stuff.StaticMethods import StaticMethods
from pathlib import Path
import pandas as pd
import seaborn as sns


class CachePrinter:
    def __init__(self, dir: Path):
        self.dir: Path = dir
        self.pool: RestartingPoolReplacement = RestartingPoolReplacement(8)

    def run(self, dir: Path = None, depth: int = 0):
        dir = dir or self.dir
        for f in dir.iterdir():
            f: Path = f
            if f.is_dir():
                self.run(f, depth=depth + 1)
                continue
            if f.name.endswith('history.csv'):
                print(f)
                # self.pool.apply_async(CachePrinter.print_f, args=(f,))
                self.print_f(f)
        if depth == 0:
            self.pool.join()

    @staticmethod
    def default_fig(w: int = 5, h: int = 4, h_offset: int = 0):
        no_rows: int = 2
        no_columns: int = 2
        fig = plt.figure()
        fig.set_figheight(no_rows * h + h_offset)
        fig.set_figwidth(no_columns * w)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

        return fig, [ax1, ax2, ax3]

    @staticmethod
    def print_f(f: Path):
        def print_simple_line_diag(df: pd.DataFrame, ax, ax_log, print_log=False):
            try:
                index = 0
                if len(df) > 1:
                    index = 1
                t: pd.DataFrame = df.iloc[index:]
                max_y = t.values.max()
                min_y = df.values.min()
                # print(f"limits {min_y} {max_y}")
                ax.set_ylim(min_y, max_y)
                ax.set_title(f"{', '.join(df.columns)}")
                sns.lineplot(data=df, ax=ax)
                if print_log:
                    df = df.copy()
                    for c in df.columns:
                        df[c] = np.log(df[c])
                    t: pd.DataFrame = df.iloc[index:]
                    max_y = t.values.max()
                    min_y = df.values.min()
                    ax_log.set_ylim(min_y, max_y)
                    ax_log.set_title(f"log {', '.join(df.columns)}")
                    sns.lineplot(data=df, ax=ax_log)
            except BaseException as e:
                print(f"{e} in '{f}'", file=sys.stderr)

        print(f"printing '{f}'")

        # df = pd.read_csv(f, index_col='epoch')
        # if 'Unnamed: 0' in df.columns:
        #     df = df.drop('Unnamed: 0', axis=1)
        # if 'js' in df.columns:
        #     df = df.drop('js', axis=1)
        # has_kl = 'kl' in df.columns
        # filtered = df[df['kl'].notnull()] if has_kl else df
        # df_loss = df.drop(['kl'], axis=1) if has_kl else df
        #
        # no_columns = 2 if has_kl else 1
        # fig, axs = CachePrinter.default_fig(w=10, h=7)
        #
        # print_simple_line_diag(df_loss, ax=axs[0], ax_log=axs[0], print_log=False)
        # if has_kl:
        #     df_kl = filtered.drop(['loss', 'val_loss'], axis=1)
        #     print_simple_line_diag(df_kl, ax=axs[1], ax_log=axs[2], print_log=True)
        # fig.tight_layout()
        # plt.savefig(f.with_name(f"{f.name}.png"))

        # look for other histories with the same name

        str_end = '.maf.history.csv'
        if f.name.endswith(str_end):
            name = f.name[:-len(str_end)]
            index = int(name[-1])
            if index == 0:
                name_start: str = name[:-1]
                d = f.parent
                related_csvs: List[Path] = []
                for o in d.iterdir():
                    if o.name.startswith(name_start) and o.name.endswith(str_end):
                        related_csvs.append(o)
                        print(f"related '{o}'")
                if len(related_csvs) < 2:
                    return
                merged_csv: pd.DataFrame = CachePrinter.merge_csvs(csvs=related_csvs)
                CachePrinter.print_related(df=merged_csv, target=Path(d, f"{name_start}.merged.png"))
        return 1

    @staticmethod
    def print_related(df: pd.DataFrame, target: Path):
        plt.clf()
        fig, axs = CachePrinter.default_fig(w=10, h=7)
        df_loss = df.loc[df['type'] != 'kl']
        df_kl = df.loc[df['type'] == 'kl'].dropna()
        df_log_kl = df_kl.copy()
        df_log_kl['value'] = np.log(df_log_kl['value'])

        axs[0].set(ylabel='Loss, Val_Loss', title='Loss and val loss over epochs')
        sns.lineplot(data=df_loss, x='epoch', y='value', hue='type', ax=axs[0])
        sns.lineplot(data=df_kl, x='epoch', y='value', ax=axs[1], ci='sd', err_style='bars')
        sns.lineplot(data=df_log_kl, x='epoch', y='value', ax=axs[2], ci='sd', err_style='bars')
        plt.tight_layout()
        plt.savefig(target)

        def print_simple_line_diag(df: pd.DataFrame, ax, ax_log, print_log=False):
            try:
                index = 0
                if len(df) > 1:
                    index = 1
                t: pd.DataFrame = df.iloc[index:]
                max_y = t.values.max()
                min_y = df.values.min()
                # print(f"limits {min_y} {max_y}")
                ax.set_ylim(min_y, max_y)
                ax.set_title(f"{', '.join(df.columns)}")
                sns.lineplot(data=df, ax=ax)
            except BaseException as e:
                print(f"{e}'", file=sys.stderr)

        print('dada')

    @staticmethod
    def merge_csvs(csvs: List[Path]) -> pd.DataFrame:
        columns: Optional[str] = None
        dfs: List[pd.DataFrame] = []
        for csv in csvs:
            df: pd.DataFrame = pd.read_csv(csv)
            if columns is None:
                columns = list(df.columns)
            dfs.append(df)
        max_len = max([len(df) for df in dfs])
        values = np.empty((len(dfs) * max_len, len(columns) + 1))
        values[:] = np.NaN
        for i, df in enumerate(dfs):
            p = i * max_len
            q = p + len(df)
            values[p: q, 1:] = df.values
            values[p:q, 0] = float(i)

        filtered = values[np.logical_not(np.all(np.isnan(values), axis=1))]
        df = pd.DataFrame(filtered, columns=['nf'] + columns)
        df = df.melt(id_vars=['nf', 'epoch'], var_name='type')
        return df


if __name__ == '__main__':
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument('--dir', help='which dir to traverse', default='.cache_test', type=str)
    args: Dict[str, Any] = vars(ap.parse_args())
    c = CachePrinter(Path(args['dir']))
    c.run()
