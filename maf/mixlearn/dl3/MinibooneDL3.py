from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from common import jsonloader
from common.NotProvided import NotProvided
from maf.DS import DatasetProps, DS
from maf.DL import DL3, DL2, DataSource
from maf.stuff.StaticMethods import StaticMethods
from maf.mixlearn.MixLearnExperiment import DatasetFetcher


class MinibooneDL3(DL3):
    def __init__(self, dataset_name: str = 'miniboone'):
        super().__init__(url='https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt',
                         dl_folder=Path(StaticMethods.cache_dir(), NotProvided.value_if_not_provided(dataset_name, 'miniboone')))
        self.cache_txt_file: Path = Path(StaticMethods.cache_dir(), 'miniboone.txt')
        self.cache_np_file: Path = Path(StaticMethods.cache_dir(), 'miniboone.npy')
        self.props_file: Path = Path(StaticMethods.cache_dir(), 'miniboone.props.json')
        self.props: Optional[DatasetProps] = None
        self.paper_load: bool = False

    def fetch_impl(self):
        if self.cache_np_file.exists():
            data = np.load(self.cache_np_file, allow_pickle=True)
            self.props = jsonloader.load_json(self.props_file)
        else:
            fetcher = DatasetFetcher(dataset_url=self.url, target_file=self.cache_txt_file)
            fetcher.fetch()

            """stolen from 'https://github.com/daines-analytics/deep-learning-projects/tree/master/py_tensorflow_binaryclass_miniboone_particle_identification"""
            signal_rec = 36499
            widthVector = [14] * 50
            colNames = ['particle' + str(i).zfill(2) for i in range(1, 51)]
            df = pd.read_fwf(self.cache_txt_file, widths=widthVector, header=None, names=colNames, skiprows=1, index_col=False, na_values=[-999], dtype=np.float32)
            df.insert(0, 'signal', 0.0)
            df.iloc[:signal_rec, 0] = 1.0
            df['signal'] = df['signal'].astype(np.float32)
            """end of theft"""
            df = df.dropna()
            data = df.values

            # following lines seem to be wrong though it is not obvious why,
            # since the data set description does not mention -999 representing NaN

            # with open(self.cache_txt_file, mode='r') as f:
            #     text = f.read()
            # print(f"parsing '{self.cache_txt_file}'")
            # """Data Set Information:
            #     The submitted file is set up as follows. In the first line is the the number of signal events
            #     followed by the number of background events.
            #     The signal events come first, followed by the background events. Each line,
            #     after the first line has the 50 particle ID variables for one event"""
            # lines = text.split('\n')
            # amounts: Tuple[str, str] = lines[0].lstrip().split(' ')
            # amounts: Tuple[int, int] = [int(s) for s in amounts]
            # no_of_signals = amounts[0]
            # no_of_noise = amounts[1]
            #
            # # parse signals
            # def parse(lines: List[str], start_index: int, amount: int) -> np.ndarray:
            #     result: List[List[float]] = []
            #     for line in lines[start_index: start_index + amount]:
            #         values = [float(s) for s in line.lstrip().replace('  ', ' ').split(' ')]
            #         result.append(values)
            #     return np.array(result, dtype=np.float32)
            #
            # signals = parse(lines, start_index=1, amount=no_of_signals)
            # noise = parse(lines, start_index=no_of_signals + 1, amount=no_of_noise)
            # ones = np.ones((signals.shape[0], 1), dtype=np.float32)
            # zeros = np.zeros((noise.shape[0], 1), dtype=np.float32)
            # data = np.concatenate([np.column_stack([ones, signals]), np.column_stack([zeros, noise])])
            np.save(self.cache_np_file, data, allow_pickle=True)

        signals = data[np.where(data[:, 0] > 0)]
        noise = data[np.where(data[:, 0] < 1)]
        self.no_of_signals = len(signals)
        self.no_of_noise = len(noise)
        self.props = DatasetProps()
        self.props.shape = data.shape
        self.props.no_of_noise = self.no_of_noise
        self.props.no_of_signals = self.no_of_signals
        self.props.classes = [1, 0]
        self.props.length = len(data)
        self.props.no_of_columns = data.shape[1]
        jsonloader.to_json(self.props, file=self.props_file, pretty_print=True)



        if self.paper_load:
            """ from: https://github.com/gpapamak/maf/blob/master/datasets/miniboone.py
                # NOTE: To remember how the pre-processing was done.
                # data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
                # print data.head()
                # data = data.as_matrix()
                # # Remove some random outliers
                # indices = (data[:, 0] < -100)
                # data = data[~indices]
                #
                # i = 0
                # # Remove any features that have too many re-occuring real values.
                # features_to_remove = []
                # for feature in data.T:
                #     c = Counter(feature)
                #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
                #     if max_count > 5:
                #         features_to_remove.append(i)
                #     i += 1
                # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
                # np.save("~/data/miniboone/data.npy", data)
            """

            def filter_paper_mode(data: np.ndarray, features_to_remove: List[int] = None) -> [np.ndarray, List[int]]:
                indices = (data[:, 0] < -100)
                data: np.ndarray = data[~indices]
                i = 0
                calc_features = False
                if features_to_remove is None:
                    calc_features = True
                    features_to_remove = []
                for feature in data.T:
                    from future.backports import Counter
                    c = Counter(feature)
                    # c = FeatureCounter(feature)
                    max_count = np.array([v for k, v in sorted(c.items())])[0]
                    if max_count > 5 and calc_features:
                        features_to_remove.append(i)
                    i += 1
                data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
                return data, features_to_remove

            data_signal, features_removed = filter_paper_mode(signals[:, 1:])
            train, test = train_test_split(data_signal, test_size=0.1, stratify=data[:, 0])
            train, val = train_test_split(train, test_size=0.1, stratify=data[:, 0])
            ds_train = tf.data.Dataset.from_tensor_slices(train)
            ds_val = tf.data.Dataset.from_tensor_slices(val)
            return ds_train, ds_val
        else:
            # split
            data: np.ndarray = np.concatenate([signals, noise], axis=0)
            # normalise
            normalised, mean, std = StaticMethods.norm(data)
            normalised_signal = normalised[:len(signals), 1:]
            normalised_noise = normalised[len(signals):, 1:]
            normalised_signal = DS.from_tensor_slices(normalised_signal)
            normalised_noise = DS.from_tensor_slices(normalised_noise)
            dl = DL2(dataset_name='miniboone',
                     dir=self.dl_folder,
                     signal_source=DataSource(ds=normalised_signal),
                     noise_source=DataSource(ds=normalised_noise),
                     amount_of_signals=len(signals),
                     amount_of_noise=len(noise))
            dl.create_data()
        print('parsing done')
