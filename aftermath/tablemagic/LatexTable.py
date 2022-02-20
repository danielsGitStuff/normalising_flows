import argparse
from pathlib import Path
from typing import List, Dict, Any

from aftermath.Constants import Constants
from common.NotProvided import NotProvided
import pandas as pd


class LatexTable:
    def __init__(self, directory: Path = NotProvided()):
        self.directory: Path = directory
        self.experiment_names: Dict[str, str] = Constants.get_name_dict()

    def run(self):
        result_dirs: List[Path] = [o for o in self.directory.iterdir() if o.is_dir() and o.name.startswith('results_')]
        for d in result_dirs:
            self.search(d, d.name)

    def search(self, directory: Path, name: str):
        for o in directory.iterdir():
            o: Path = o
            if o.name == f"{name}.csv":
                self.print_latex(o)

    def print_latex(self, csv_f: Path):
        df: pd.DataFrame = pd.read_csv(csv_f)

        related_dfs: List[pd.DataFrame] = []
        related_names: List[str] = []
        for name in df["Name"]:
            f = Path(csv_f.parent, f"{name}.tablemagic.csv")
            if f.exists():
                related_dfs.append(pd.read_csv(f))
                related_names.append(name)

        names: List[str] = []
        for name, d in zip(related_names, related_dfs):
            names += [name] * len(d)
        ddd = pd.concat(related_dfs)
        ddd['name'] = names
        ddd = ddd.drop(['val_loss'], axis=1)
        ddd = ddd.sort_values(['layers', 'name'], ascending=False)
        # print(df)
        lines: List[str] = ['\begin{table}[H]',
                            '\label{table:Dim10}']
        # \begin{tabular}{@{}l|llll|llll@{}}
        s = '\begin{tabular}{@{}l'
        for row in df['Name'].values:
            s += '|llll'
        s += '@{}}'
        lines.append(s)
        # & \multicolumn{4}{c}{\bm{Large Gaps}}  &         \multicolumn{4}{c}{\bm{Small Gaps}} &   \\
        s = '& \multicolumn{4}{c}'
        names = [self.experiment_names.get(name, name) for name in related_names]
        name_index_map: Dict[str, int] = {name: i for i, name in enumerate(related_names)}
        for name in names:
            s += '{\bm{' + name + '}}  & '
        s += '\\'
        lines.append(s)
        # Layers     & Epoch & KL & Avg(KL) & Epoch      & KL & Avg(KL)\\
        s = 'Layers '
        for name in names:
            s += '& Epoch & KL & Avg(KL)'
        s += '\\'
        lines.append(s)
        lines.append('\hline')
        unique_layers = set(ddd['layers'].unique())
        for layers in unique_layers:
            d: pd.DataFrame = ddd.loc[ddd['layers'] == layers]
            line = [' &'] * len(related_names) * 3
            line = line + ['\\']
            max_entries = 0
            for n in d['name'].unique():
                max_entries = max(max_entries, len(d.loc[d['name'] == n]))
            block = [line] * max_entries
            for name in d['name'].unique():
                dd: pd.DataFrame = d.loc[d['name'] == name]
                name_index = name_index_map[name]
                i = 0
                for _, row in dd.iterrows():
                    block[i][name_index] = f"{row['epoch']} &"
                    block[i][name_index + 1] = f"{row['kl']} &"
                    i += 1
        bblock = [' '.join(line) for line in block]
        print(lines)


if __name__ == '__main__':
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument('--dir', help='which dir to traverse', default='pull/', type=str)
    args: Dict[str, Any] = vars(ap.parse_args())
    t = LatexTable(Path(args['dir']))
    t.run()
