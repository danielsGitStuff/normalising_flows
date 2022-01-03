import shutil
from pathlib import Path

import numpy as np
from scipy.stats import zscore

from common.NotProvided import NotProvided
from distributions.base import cast_dataset_to_tensor, cast_to_ndarray
from maf.DS import DS
from maf.DL import DL2, DataSource
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
import pandas as pd


class DSOutlierInitProcess(DSInitProcess):
    def __init__(self, dl_cache_dir: Path = NotProvided(), experiment_cache_dir: Path = NotProvided(), test_split: float = 0.1):
        super().__init__(dl_cache_dir=dl_cache_dir, experiment_cache_dir=experiment_cache_dir, test_split=test_split)

    def after_initialisation(self):
        print('incoming')
        testing_file = Path(self.train_dir, 'testing_after_init.executed')
        if testing_file.exists():
            print('removing outliers omitted because it was already done. Have a good day Sir!')
            return
        max_deviation = 3.0
        dl = DL2.load(self.train_dir)
        signals: DS = dl.get_signal(dl.props.no_of_signals)
        noise: DS = dl.get_noise(dl.props.no_of_noise)
        signals, _ = cast_dataset_to_tensor(signals)
        noise, _ = cast_dataset_to_tensor(noise)
        signals: np.ndarray = cast_to_ndarray(signals)
        df: pd.DataFrame = pd.DataFrame(signals)
        z_scores = zscore(df)
        filtered = (z_scores < max_deviation).all(axis=1)
        filtered_df = df[filtered]
        means = np.mean(signals)
        std = np.std(signals)
        print(f"removing {len(signals) - len(filtered_df)} outlier signals with more than {max_deviation} std deviations from '{self.train_dir}'")
        ds_signal = DS.from_tensor_slices(filtered_df.values)
        ds_noise = DS.from_tensor_slices(noise)
        shutil.rmtree(self.train_dir)
        dl_new = DL2(dataset_name=dl.dataset_name,
                     dir=self.train_dir,
                     signal_source=DataSource(ds=ds_signal),
                     noise_source=DataSource(ds=ds_noise),
                     amount_of_noise=dl.props.no_of_noise,
                     amount_of_signals=len(ds_signal))
        dl_new.create_data()
        testing_file.touch()
        print('modifying train data set done')
        # print('###### ENDS HERE ######')
        # print('')
        # print('Outlier samples have been stripped off the training dataset.')
        # print('Need to restart the program to continue')
        # sys.exit(0)
