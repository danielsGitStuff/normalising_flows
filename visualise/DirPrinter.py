import numpy as np
from matplotlib import pyplot as plt

from common import jsonloader
from common.globals import Global
from common.jsonloader import Ser, SerSettings
from common.poolreplacement import RestartingPoolReplacement
from maf.stuff.StaticMethods import StaticMethods
from typing import Dict, Any

import argparse
from pathlib import Path
from visualise.CachePrinter import CachePrinter
import pandas as pd
import seaborn as sns


class DirPrinter(Ser):
    @staticmethod
    def static_run_cache_printer(js: str):
        c: CachePrinter = jsonloader.from_json(js)
        c.run()

    @staticmethod
    def static_print_divergences_csv(f: Path):
        df: pd.DataFrame = pd.read_csv(f)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df = df.astype({'layers': int})
        plt.clf()

        # big = 32
        # medium = 28
        # small = 24
        # plt.rc('font', size=small)
        # plt.rc('axes', titlesize=small)
        # plt.rc('axes', labelsize=medium)
        # plt.rc('xtick', labelsize=small)
        # plt.rc('ytick', labelsize=small)
        # plt.rc('legend', fontsize=small)
        # plt.rc('figure', titlesize=big)
        # plt.rc('lines', linewidth=3)

        fig, axs = StaticMethods.default_fig(2, 1, w=7, h=5)
        axs = axs.flatten()
        df_log = df.copy()
        df_log['kl'] = np.log(df['kl'])
        cmap = sns.color_palette("flare")
        sns.barplot(data=df, x='layers', y='kl', ax=axs[0], palette=cmap, ci='sd')
        sns.barplot(data=df_log, x='layers', y='kl', ax=axs[1], palette=cmap, ci='sd')
        axs[0].set(ylabel='KL', xlabel='Layers')
        axs[1].set(ylabel='log(KL)', xlabel='Layer')
        plt.tight_layout()
        plt.savefig(Path(f.parent, f"{f.name}.png"))

    def run_cache_printer(self, d: Path):
        c = CachePrinter(d)
        js = jsonloader.to_json(c)
        self.pool.apply_async(DirPrinter.static_run_cache_printer, args=(js,))

    def __init__(self, directory: Path):
        super().__init__()
        self.directory: Path = directory
        self.pool: RestartingPoolReplacement = RestartingPoolReplacement(4)

    def run(self):
        for o in self.directory.iterdir():
            o: Path = o
            if o.is_dir():
                if o.name == '.cache':
                    # self.run_cache_printer(o)
                    print('cache')
                elif o.name.startswith('result'):
                    for f in o.iterdir():
                        f: Path = f
                        if f.is_file() and f.name.endswith('.divergences.csv'):
                            self.pool.apply_async(DirPrinter.static_print_divergences_csv, args=(f,))
                            # DirPrinter.static_print_divergences_csv(f)
        self.pool.join()


if __name__ == '__main__':
    SerSettings.enable_testing_mode()
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument('--dir', help='which dir to traverse', default='pull.olde2/', type=str)
    args: Dict[str, Any] = vars(ap.parse_args())
    d = DirPrinter(Path(args['dir']))
    d.run()
