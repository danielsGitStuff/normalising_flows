from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import BinaryCrossentropy

from common import jsonloader
from common.NotProvided import NotProvided
from common.jsonloader import Ser
from distributions.base import enable_memory_growth, cast_dataset_to_tensor
from common.globals import Global
from keta.lazymodel import LazyModel
from maf.DS import DS
from maf.DL import DL2

from maf.mixlearn.KerasETA import KerasETA


class BinaryClassifierCreator(Ser):
    def __init__(self):
        super().__init__()

    def create_classifier(self, input_dims: int) -> LazyModel:
        raise NotImplementedError()


class ClassifierTrainingProcess(Ser):
    def __init__(self,
                 dl_training_genuine: DL2 = NotProvided(),
                 dl_training_synth: DL2 = NotProvided(),
                 dl_val_genuine: DL2 = NotProvided(),
                 dl_val_synth: DL2 = NotProvided(),
                 dl_test: DL2 = NotProvided(),

                 history_csv_file: Path = NotProvided(),
                 ds_training_folder: Path = NotProvided(),
                 ds_synth_training_folder: Path = NotProvided(),
                 ds_synth_val_folder: Path = NotProvided(),
                 ds_val_folder: Path = NotProvided(),
                 ds_test_folder: Path = NotProvided(),
                 epochs: int = NotProvided(),
                 clf_t_ge_noi: int = NotProvided(),
                 clf_t_ge_sig: int = NotProvided(),
                 clf_t_sy_noi: int = NotProvided(),
                 clf_t_sy_sig: int = NotProvided(),
                 clf_v_ge_noi: int = NotProvided(),
                 clf_v_ge_sig: int = NotProvided(),
                 clf_v_sy_noi: int = NotProvided(),
                 clf_v_sy_sig: int = NotProvided(),

                 # clf_t_g_size: int = NotProvided(),
                 # clf_t_s_size: int = NotProvided(),
                 # clf_v_g_size: int = NotProvided(),
                 # clf_v_s_size: int = NotProvided(),
                 model_base_file: str = NotProvided(),
                 conditional_dims: int = 0,
                 batch_size: Optional[int] = None):
        super().__init__()
        self.batch_size: Optional[int] = batch_size
        self.conditional_dims: int = conditional_dims
        self.dl_training_genuine: DL2 = dl_training_genuine
        self.dl_training_synth: DL2 = dl_training_synth
        self.dl_val_genuine: DL2 = dl_val_genuine
        self.dl_val_synth: DL2 = dl_val_synth
        self.dl_test: DL2 = dl_test
        # self.ds_synth_training_folder: Path = ds_synth_training_folder
        # self.ds_synth_val_folder: Path = ds_synth_val_folder
        # self.ds_test_folder: Path = ds_test_folder
        # self.ds_training_folder: Path = ds_training_folder
        # self.ds_val_folder: Path = ds_val_folder
        self.epochs: int = epochs
        self.history_csv_file: Path = history_csv_file
        self.model_base_file: str = model_base_file
        # self.clf_t_g_size: int = clf_t_g_size
        # self.clf_t_s_size: int = clf_t_s_size
        # self.clf_v_g_size: int = clf_v_g_size
        # self.clf_v_s_size: int = clf_v_s_size

        self.clf_t_ge_noi: int = clf_t_ge_noi
        self.clf_t_ge_sig: int = clf_t_ge_sig
        self.clf_t_sy_noi: int = clf_t_sy_noi
        self.clf_t_sy_sig: int = clf_t_sy_sig
        self.clf_v_ge_noi: int = clf_v_ge_noi
        self.clf_v_ge_sig: int = clf_v_ge_sig
        self.clf_v_sy_noi: int = clf_v_sy_noi
        self.clf_v_sy_sig: int = clf_v_sy_sig

    def create_classifier(self) -> LazyModel:
        input_dim = self.dl_training_genuine.props.dimensions
        ins = Input(shape=(input_dim,))
        b = Dense(512, activation='relu')(ins)
        # b = Dropout()
        b = Dense(512, activation='relu')(b)
        b = Dense(1, activation='sigmoid')(b)
        # b = Dense(100, activation='relu', name='DenseRELU0')(ins)
        # b = Dense(100, activation='relu')(b)
        # b = BatchNormalization()(b)
        # b = Dense(100, activation='relu')(b)
        # b = Dense(1, activation='linear', name='out')(b)
        model = Model(inputs=[ins], outputs=[b])
        lm = LazyModel.Methods.wrap(model)
        lm.compile(optimizer='adam', loss=BinaryCrossentropy(), lr=0.001, metrics=['accuracy'])
        return lm

    def run(self) -> Dict[str, float]:
        # todo continue training broken
        # the code depends on an EarlyStop object which is of course non existent when the model was loaded.

        def prepare(genuine: DL2, synth: DL2, clf_ge_sig: int, clf_ge_no: int, clf_sy_sig: int, clf_sy_no: int) -> DS:
            gen = genuine.get_conditional(amount_signal=clf_ge_sig, amount_noise=clf_ge_no)
            synth = synth.get_conditional(amount_signal=clf_sy_sig, amount_noise=clf_sy_no)
            return gen.concatenate(synth)

        import tensorflow as tf

        ds_train = prepare(genuine=self.dl_training_genuine, synth=self.dl_training_synth,
                           clf_ge_sig=self.clf_t_ge_sig,
                           clf_ge_no=self.clf_t_ge_noi,
                           clf_sy_sig=self.clf_t_sy_sig,
                           clf_sy_no=self.clf_t_sy_noi)
        ds_val = prepare(genuine=self.dl_val_genuine, synth=self.dl_val_synth,
                         clf_ge_sig=self.clf_v_ge_sig,
                         clf_ge_no=self.clf_v_ge_noi,
                         clf_sy_sig=self.clf_v_sy_sig,
                         clf_sy_no=self.clf_v_sy_noi)

        epoch = None
        es = None
        loaded = False
        self.epochs = 10
        if LazyModel.Methods.model_exists(self.model_base_file):
            lm = LazyModel.Methods.load_from_file(self.model_base_file)
            loaded = True
            h_pd: pd.DataFrame = pd.read_csv(self.history_csv_file, index_col=False)
            epoch = int(h_pd['epoch'].max())
        else:
            lm = self.create_classifier()

            print(
                f"fitting classifier with {len(ds_train)} samples (clf_t_ge_sig {self.clf_t_ge_sig} clf_t_ge_noi {self.clf_t_ge_noi} clf_t_sy_sig {self.clf_t_sy_sig} clf_t_sy_noi {self.clf_t_sy_noi}) -> '{self.history_csv_file}'")
            es = EarlyStopping(monitor='val_loss', patience=15, verbose=0, restore_best_weights=True, mode='min')
            if len(ds_train) > 0:
                epochs = Global.Testing.get('testing_epochs', self.epochs)
                eta = KerasETA(interval=10, epochs=epochs)
                history = lm.fit_data_set(ds_train, conditional_dims=self.conditional_dims, ds_val=ds_val, batch_size=self.batch_size, epochs=epochs, callbacks=[es, eta],
                                          shuffle=True)
                epoch = epochs
                if es.best_epoch is not None and es.best_epoch != epochs:
                    epoch = es.best_epoch
            else:
                history = lambda: None
                history.epoch = -1
                history.history = {'accuracy': [-1.0], 'loss': [-1.0], 'val_loss': [-1.0]}
            lm.base_file = self.model_base_file
            lm.save()
        if not loaded:
            h_d: Dict[str, List[float]] = history.history.copy()
            h_d['epoch'] = history.epoch
            h_pd: pd.DataFrame = pd.DataFrame.from_dict(h_d)
            # cut off stuff that is beyond es.best_epoch
            columns = list(h_pd.columns)
            vs: np.ndarray = h_pd.values
            if hasattr(es, 'best_epoch'):
                vs = vs[:es.best_epoch + 1]
            h_pd: pd.DataFrame = pd.DataFrame(vs, columns=columns)
            h_pd.to_csv(self.history_csv_file)
            del ds_val, ds_train

        ds_test = self.dl_test.get_conditional()
        # ds_test = tf.data.experimental.load(str(self.ds_test_folder))
        vs, ms = lm.evaluate_data_set(ds_test, conditional_dims=self.conditional_dims)

        truth = ds_test.map(lambda t: t[0])
        truth, _ = cast_dataset_to_tensor(truth)
        ps = lm.predict_data_set(ds_test, conditional_dims=self.conditional_dims, batch_size=self.batch_size)
        ps = np.round(ps)
        cm = tf.math.confusion_matrix(truth, ps, num_classes=2)
        d: Dict[str, float] = {m: v for m, v in zip(ms, vs)}
        if epoch is not None:
            d['max_epoch'] = epoch
        d['tnoise'] = float(cm[0][0])
        d['fnoise'] = float(cm[1][0])
        d['fsig'] = float(cm[0][1])
        d['tsig'] = float(cm[1][1])
        return d

    @staticmethod
    def static_execute(js: str) -> Dict[str, float]:
        """train classifier, return TEST results"""
        print('running in process')
        enable_memory_growth()
        classifier: ClassifierTrainingProcess = jsonloader.from_json(js)
        result = classifier.run()
        print('done with process')
        return result

    def execute(self) -> Dict[str, float]:
        # return self.run()
        js = jsonloader.to_json(self, pretty_print=True)
        # print('debug skip pool')
        # return ClassifierTrainingProcess.static_execute(js)
        res = Global.POOL().run_blocking(ClassifierTrainingProcess.static_execute, args=(js,))
        return res
