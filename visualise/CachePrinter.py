import numpy as np
from matplotlib import pyplot as plt

from maf.stuff.StaticMethods import StaticMethods
from pathlib import Path
import pandas as pd
import seaborn as sns


class CachePrinter:
    def __init__(self, dir: Path):
        self.dir: Path = dir

    def run(self):
        for f in self.dir.iterdir():
            f: Path = f
            if not f.is_file():
                continue
            if f.name.endswith('maf.history.csv'):
                self.print_f(f)

    def print_f(self, f: Path):
        fig, axs = StaticMethods.default_fig(1, 2, w=10, h=7)
        df = pd.read_csv(f, index_col=False)
        filtered = df[df['kl'].notnull()]
        df_kl = filtered.drop(['epoch', 'loss', 'val_loss'], axis=1)
        df_kl['kl'] = np.log(df_kl['kl'])
        df_kl['js'] = np.log(df_kl['js'])
        # filtered = filtered.pivot('epoch','kl')
        print(df_kl)
        sns.lineplot(data=df_kl, ax=axs[0])

        df_loss = df.drop(['epoch', 'kl', 'js'], axis=1)
        # df_loss['loss'] = np.log(df_loss['loss'])
        # df_loss['val_loss'] = np.log(df_loss['val_loss'])
        ax = axs[1]
        t: pd.DataFrame = df_loss.iloc[10:]
        max_y = max(t['loss'].max(), t['val_loss'].max())
        min_y = min(t['loss'].min(), t['val_loss'].min())
        ax.set_ylim(min_y, max_y)

        # print every 10th or so
        idx = [i for i in df_loss.index.values if i % 10 == 0]
        df_loss = df_loss.iloc[idx]

        sns.lineplot(data=df_loss, ax=ax)
        fig.tight_layout()
        plt.savefig(f.with_name(f"{f.name}.png"))


if __name__ == '__main__':
    c = CachePrinter(Path('.cache'))
    c.run()
