from __future__ import annotations
from pathlib import Path
from typing import List, Type

from keras import Input, Model
from keras.layers import Dense
from keras.losses import BinaryCrossentropy

from keta.lazymodel import LazyModel
from maf.DL import DL2
from maf.MaskedAutoregressiveFlow import MAFCreator
from distributions.LearnedDistribution import LearnedDistributionCreator
from maf.mixlearn.ClassifierTrainingProcess import BinaryClassifierCreator
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
from maf.mixlearn.dl3.MinibooneDL3 import MinibooneDL3
from maf.mixlearn.MixLearnExperiment import MixLearnExperiment
from maf.variable.TrainingPlanner import TrainingPlanner
from maf.variable.VariableParam import FixedParam, LambdaParams, VariableParam, VariableParamInt, MetricParam


class MiniBooneBinaryClassifierCreator(BinaryClassifierCreator):
    def __init__(self):
        super().__init__()

    def create_classifier(self, input_dims: int) -> LazyModel:
        ins = Input(shape=(input_dims,))
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


class MixLearnExperimentMiniBoone(MixLearnExperiment):
    def __init__(self, name: str,
                 learned_distr_creator: LearnedDistributionCreator,
                 dataset_name: str,
                 result_folder: Path,
                 epochs: int,
                 batch_size: int = 128,
                 paper_load: bool = False,
                 experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess,
                 test_split: float = 0.1):
        self.paper_load: bool = paper_load
        self.dataset_name: str = dataset_name

        super().__init__(name=name,
                         epochs=epochs,
                         # layers=layers,
                         batch_size=batch_size,
                         # hidden_shape=hidden_shape,c
                         # norm_layer=norm_layer,
                         # noise_variance=noise_variance,
                         # batch_norm=batch_norm,
                         # use_tanh_made=use_tanh_made,
                         test_split=test_split,
                         result_folder=result_folder,
                         experiment_init_ds_class=experiment_init_ds_class,
                         learned_distribution_creator=learned_distr_creator,
                         classifier_creator=MiniBooneBinaryClassifierCreator())

    def print_divergences(self):
        pass

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(self.cache_dir, "miniboone_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def create_data_loader(self, norm_data: bool) -> DL2:
        dl3 = MinibooneDL3(dataset_name=self.dataset_name)
        return dl3.execute()

    def _run(self):
        self.start()


    @staticmethod
    def main_static(dataset_name: str, experiment_name: str, learned_distr_creator: LearnedDistributionCreator,
                    experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess):
        result_folder = Path('results')

        m = MixLearnExperimentMiniBoone(name=experiment_name,
                                        learned_distr_creator=learned_distr_creator,
                                        dataset_name=dataset_name,
                                        result_folder=result_folder,
                                        paper_load=False,
                                        epochs=100,
                                        batch_size=1000,
                                        dataset_size_steps=7,
                                        synth_ratio_steps=7,
                                        # layers=20,
                                        # hidden_shape=[200, 200],
                                        # norm_layer=False,
                                        # noise_variance=0.0,
                                        # batch_norm=True,
                                        # use_tanh_made=True,
                                        experiment_init_ds_class=experiment_init_ds_class)
        m.create_training_plan()
        m.start()
