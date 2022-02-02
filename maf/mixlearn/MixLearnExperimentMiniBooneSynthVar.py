from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone
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


class MixLearnExperimentMiniBooneSynthVar(MixLearnExperimentMiniBoone):
    def __init__(self, name: str,
                 learned_distr_creator: LearnedDistributionCreator,
                 dataset_name: str,
                 result_folder: Path,
                 epochs: int,
                 batch_size: int = 128,
                 paper_load: bool = False,
                 experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess,
                 test_split: float = 0.1):
        super().__init__(
            name=name,
            learned_distr_creator=learned_distr_creator,
            dataset_name=dataset_name,
            result_folder=result_folder,
            clf_epochs=epochs,
            batch_size=batch_size,
            paper_load=paper_load,
            experiment_init_ds_class=experiment_init_ds_class,
            test_split=test_split)

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(self.cache_dir, "miniboone_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def create_data_loader(self, norm_data: bool) -> DL2:
        dl3 = MinibooneDL3(dataset_name=self.dataset_name)
        return dl3.execute()

    def _create_training_plan(self) :
        # self.training_planner = TrainingPlanner(FixedParam('done', 0),
        #                                         LambdaParams.tsize_from_dsize(val_size=self.val_size),
        #                                         LambdaParams.vsize_from_dsize(val_size=self.val_size),
        #                                         VariableParam('dsize', range_start=self.dataset_size_start, range_end=self.dataset_size_end, range_steps=self.dataset_size_steps),
        #                                         VariableParam('synthratio', range_start=self.synth_ratio_start, range_end=self.synth_ratio_end, range_steps=self.synth_ratio_steps),
        #                                         VariableParamInt('model', range_start=0, range_end=self.classifiers_per_nf, range_steps=self.classifiers_per_nf),
        #                                         MetricParam('loss'),
        #                                         MetricParam('accuracy'),
        #                                         MetricParam('max_epoch'),
        #                                         MetricParam('tnoise'),
        #                                         MetricParam('fnoise'),
        #                                         MetricParam('tsig'),
        #                                         MetricParam('fsig'),
        #                                         CopyFromParam('clfsize', source_param='dsize'),
        #                                         LambdaParams.clf_t_g_size_from_clfsize_synthratio(val_size=self.val_size),
        #                                         LambdaParams.clf_t_s_size_from_clfsize_synthratio(val_size=self.val_size),
        #                                         LambdaParams.clf_v_g_size_from_clfsize_synthratio(val_size=self.val_size),
        #                                         LambdaParams.clf_v_s_size_from_clfsize_synthratio(val_size=self.val_size)) \
        #     .build_plan()
        return TrainingPlanner(FixedParam('done', 0),
                                                LambdaParams.tsize_from_dsize(val_size=self.val_size),
                                                LambdaParams.vsize_from_dsize(val_size=self.val_size),
                                                FixedParam('dsize', self.dataset_size_end),
                                                VariableParam('synthratio', range_start=0.0, range_end=1.0, range_steps=3),
                                                VariableParamInt('model', range_start=1, range_end=self.classifiers_per_nf, range_steps=self.classifiers_per_nf),
                                                MetricParam('loss'),
                                                MetricParam('accuracy'),
                                                MetricParam('max_epoch'),
                                                MetricParam('tnoise'),
                                                MetricParam('fnoise'),
                                                MetricParam('tsig'),
                                                MetricParam('fsig'),
                                                VariableParam('clfsize', range_start=2500, range_end=110000, range_steps=10),
                                                LambdaParams.clf_t_g_size_from_clfsize_synthratio(val_size=self.val_size),
                                                LambdaParams.clf_t_s_size_from_clfsize_synthratio(val_size=self.val_size),
                                                LambdaParams.clf_v_g_size_from_clfsize_synthratio(val_size=self.val_size),
                                                LambdaParams.clf_v_s_size_from_clfsize_synthratio(val_size=self.val_size))

    @staticmethod
    def main_static(dataset_name: str, experiment_name: str, learned_distr_creator: LearnedDistributionCreator, experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess):
        # enable_memory_growth()
        # print(tf.config.list_physical_devices())
        result_folder = Path('results')

        m = MixLearnExperimentMiniBoone(name=experiment_name,
                                        learned_distr_creator=learned_distr_creator,
                                        dataset_name=dataset_name,
                                        result_folder=result_folder,
                                        paper_load=False,
                                        clf_epochs=100,
                                        batch_size=1000,
                                        # layers=20,
                                        # hidden_shape=[200, 200],
                                        # norm_layer=False,
                                        # noise_variance=0.0,
                                        # batch_norm=True,
                                        # use_tanh_made=True,
                                        experiment_init_ds_class=experiment_init_ds_class)
        m.create_training_plan()
        m.start()
