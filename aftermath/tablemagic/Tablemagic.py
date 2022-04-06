from pandas import Series

import argparse
from typing import Dict, Any, List, Tuple

import re
from pathlib import Path
import pandas as pd


class TableMagic:
    class Methods:
        @staticmethod
        def get_name_from_divergences_csv(csv_f: Path) -> str:
            match_primary_name = re.search('.*(?=\.divergences\.csv$)', csv_f.name)
            name = csv_f.name[match_primary_name.start():match_primary_name.end()]
            return name

    def __init__(self, directory: Path):
        self.root: Path = directory
        self.cache: Path = Path(directory, '.cache')

    def run(self):
        for o in self.root.iterdir():
            o: Path = o
            if o.is_dir() and o.name.startswith('results_'):
                print(o)
                for f in o.iterdir():
                    f: Path = f
                    if f.is_file() and f.name.endswith('.divergences.csv'):
                        name = TableMagic.Methods.get_name_from_divergences_csv(f)
                        target_file = Path(f.parent, f"{name}.tablemagic.csv")
                        primary_df, related = self.find_related_of_divergence(f)
                        self.merge(primary_name=name, primary_df=primary_df, related=related, target_file=target_file)

    def find_related_of_divergence(self, csv_f: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        primary_df: pd.DataFrame = pd.read_csv(csv_f)
        if 'Unnamed: 0' in primary_df.columns:
            primary_df = primary_df.drop('Unnamed: 0', axis=1)
        name = TableMagic.Methods.get_name_from_divergences_csv(csv_f)
        related_dfs: Dict[str, pd.DataFrame] = {}
        for f in self.cache.iterdir():
            f: Path = f
            if f.name.startswith(name) and f.name.endswith('maf.history.csv'):
                df: pd.DataFrame = pd.read_csv(f)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop('Unnamed: 0', axis=1)
                related_dfs[f.name] = df
        return primary_df, related_dfs

    def merge(self, primary_name: str, primary_df: pd.DataFrame, related: List[pd.DataFrame], target_file: Path):
        primary_df['epoch'] = 0
        primary_df['val_loss'] = 0.0
        for index, row in primary_df.iterrows():
            row: Series = row
            # Dim10cSmallGaps_l3.13.maf.history.csv
            layers = int(row['layers'])
            related_name = f"{primary_name}_l{layers}.{index}.maf.history.csv"
            df = related[related_name]
            primary_df.at[index, 'epoch'] = int(df['epoch'].max())
            primary_df.at[index,'val_loss'] = df['val_loss'].values[-1]
        primary_df.to_csv(target_file, index=False)


if __name__ == '__main__':
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument('--dir', help='which dir to traverse', default='./', type=str)
    args: Dict[str, Any] = vars(ap.parse_args())
    t = TableMagic(Path(args['dir']))
    t.run()
