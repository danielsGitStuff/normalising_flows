import numpy as np
from matplotlib import pyplot as plt

from common.poolreplacement import RestartingPoolReplacement
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
                self.pool.apply_async(CachePrinter.print_f, args=(f,))
                # self.print_f(f)
        if depth == 0:
            self.pool.join()

    @staticmethod
    def print_f(f: Path):
        # if f.name == 'NF2D_Row3':
        #     print('debug 4gerg')

        def printi(df: pd.DataFrame, ax, ax_log, print_log=False):
            index = 0
            if len(df) > 1:
                index = 1
            t: pd.DataFrame = df.iloc[index:]
            max_y = t.values.max()
            min_y = df.values.min()
            print(f"limits {min_y} {max_y}")
            ax.set_ylim(min_y, max_y)
            ax.set_title(f"{', '.join(df.columns)}")
            sns.lineplot(data=df, ax=ax)
            if print_log:
                df = df.copy()
                for c in df.columns:
                    df[c] = np.log(df[c])
                # df['kl'] = np.log(df_kl['kl'])
                # df['js'] = np.log(df_kl['js'])
                # t: pd.DataFrame = df.loc[df.index >= 10]  # df.iloc[10:]
                t: pd.DataFrame = df.iloc[index:]
                max_y = t.values.max()
                min_y = df.values.min()
                ax_log.set_ylim(min_y, max_y)
                ax_log.set_title(f"log {', '.join(df.columns)}")
                sns.lineplot(data=df, ax=ax_log)
            else:
                ax_log.set_axis_off()

        print(f"printing '{f}'")

        df = pd.read_csv(f, index_col='epoch')
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        has_kl = 'kl' in df.columns
        filtered = df[df['kl'].notnull()] if has_kl else df
        df_loss = df.drop(['kl', 'js'], axis=1) if has_kl else df

        no_columns = 2 if has_kl else 1
        fig, axs = StaticMethods.default_fig(2, no_columns=no_columns, w=10, h=7)
        axs = axs.reshape((2, no_columns))
        printi(df_loss, ax=axs[0][0], ax_log=axs[1][0], print_log=False)
        if has_kl:
            df_kl = filtered.drop(['loss', 'val_loss'], axis=1)
            printi(df_kl, ax=axs[0][1], ax_log=axs[1][1], print_log=True)
        fig.tight_layout()
        plt.savefig(f.with_name(f"{f.name}.png"))
        return 1


if __name__ == '__main__':
    c = CachePrinter(Path('.cache'))
    c.run()
