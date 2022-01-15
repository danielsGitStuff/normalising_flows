from pathlib import Path

import numpy as np
from keras import Input, Model
from keras.layers import Dense, BatchNormalization
from keras.losses import BinaryCrossentropy

from common.globals import Global
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.LearnedDistribution import LearnedDistributionCreator
from distributions.MultimodalDistribution import MultimodalDistribution
from keta.lazymodel import LazyModel
from maf.DL import DL2, DataSource
from maf.mixlearn.ClassifierTrainingProcess import BinaryClassifierCreator
from maf.mixlearn.MixLearnExperiment import MixLearnExperiment
from maf.variable.TrainingPlanner import TrainingPlanner
from maf.variable.VariableParam import FixedParam, LambdaParams, VariableParam, VariableParamInt, MetricParam, CopyFromParam


class SynthBinaryClassifierCreator(BinaryClassifierCreator):
    def __init__(self):
        super().__init__()

    def create_classifier(self, input_dims: int) -> LazyModel:
        ins = Input(shape=(input_dims,))
        b = Dense(512, activation='relu')(ins)
        b = Dense(512, activation='relu')(b)
        b = BatchNormalization()(b)
        b = Dense(512, activation='relu')(b)
        b = Dense(1, activation='sigmoid')(b)
        model = Model(inputs=[ins], outputs=[b])
        lm = LazyModel.Methods.wrap(model)
        lm.compile(optimizer='adam', loss=BinaryCrossentropy(), lr=0.001, metrics=['accuracy'])
        return lm


class MixLeanExperimentSynth(MixLearnExperiment):
    @staticmethod
    def run_me(dataset_name: str, experiment_name: str, dim: int, creator: LearnedDistributionCreator):
        m = MixLeanExperimentSynth(name=experiment_name,
                                   result_folder=Path('results'),
                                   epochs=100,
                                   creator=creator,
                                   dataset_name=dataset_name,
                                   dim=dim)
        m.create_training_plan()
        m.start()

    def __init__(self, name: str, dataset_name: str, result_folder: Path, epochs: int, creator: LearnedDistributionCreator, dim: int):
        self.dataset_name: str = dataset_name
        self.dim = dim
        super().__init__(name=name,
                         norm_data=True,
                         result_folder=result_folder,
                         epochs=epochs,
                         learned_distribution_creator=creator,
                         batch_size=1000,
                         conditional=False,
                         sample_variance_multiplier=0.5,
                         classifier_creator=SynthBinaryClassifierCreator())
        self.synth_ratio_steps = 2
        self.dataset_size_start = 2500
        self.dataset_size_end = 50000
        self.dataset_size_steps = 2
        self.classifiers_per_nf = 1

    def create_data_loader(self, norm_data: bool) -> DL2:
        dim = self.dim
        # return DL2(dataset_name='ds_synth6',
        #            amount_of_noise=100000,
        #            amount_of_signals=100000,
        #            signal_source=DataSource(distribution=GaussianMultivariate(input_dim=dim, mus=[2] * dim, cov=[1] * dim)),
        #            noise_source=DataSource(distribution=GaussianMultivariate(input_dim=dim, mus=[0] * dim, cov=[1] * dim)))
        # signal_d = [GaussianMultivariate(input_dim=dim,
        #                                  mus=[3.0] * dim,
        #                                  cov=[1.0] * dim),
        #             GaussianMultivariate(input_dim=dim,
        #                                  mus=[-3.0] * dim,
        #                                  cov=[1.0] * dim)]
        # noise_d = [GaussianMultivariate(input_dim=dim,
        #                                 mus=[0.0] * dim,
        #                                 cov=[1.0] * dim)]
        MAX_R = 6.0
        np.random.seed(42)
        # this is easy - 3d
        amount_signal = np.random.randint(6, 10)
        signal_d = [GaussianMultivariate(input_dim=dim,
                                         mus=[np.random.uniform(-MAX_R, MAX_R) for _ in range(dim)],
                                         cov=[1.0 + np.random.uniform(0.7, 2.1) for _ in range(dim)]) for _ in range(amount_signal)]
        amount_noise = np.random.randint(6, 10)
        noise_d = [GaussianMultivariate(input_dim=dim,
                                        mus=[np.random.uniform(-MAX_R, MAX_R) for _ in range(dim)],
                                        cov=[1.0 + np.random.uniform(1.0, 3.0) for _ in range(dim)]) for _ in range(amount_noise)]
        # this is more complicated - 5d
        amount_signal = np.random.randint(7, 12)
        signal_d = [GaussianMultivariate(input_dim=dim,
                                         mus=[np.random.uniform(-MAX_R, MAX_R) for _ in range(dim)],
                                         cov=[1.0 + np.random.uniform(2.2, 4.5) for _ in range(dim)]) for _ in range(amount_signal)]
        amount_noise = np.random.randint(7, 12)
        noise_d = [GaussianMultivariate(input_dim=dim,
                                        mus=[np.random.uniform(-MAX_R, MAX_R) for _ in range(dim)],
                                        cov=[1.0 + np.random.uniform(2.0, 4.6) for _ in range(dim)]) for _ in range(amount_noise)]
        signal = DataSource(distribution=MultimodalDistribution(input_dim=dim,
                                                                distributions=signal_d))
        noise = DataSource(distribution=MultimodalDistribution(input_dim=dim,
                                                               distributions=noise_d))
        dl2 = DL2(amount_of_noise=100000,
                  amount_of_signals=100000,
                  signal_source=signal,
                  noise_source=noise,
                  dataset_name=self.dataset_name)
        return dl2

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(self.cache_dir, f"synth_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def create_training_plan(self):
        self.training_planner = TrainingPlanner(FixedParam('done', 0),
                                                LambdaParams.tsize_from_dsize(val_size=self.val_size),
                                                LambdaParams.vsize_from_dsize(val_size=self.val_size),
                                                VariableParam('dsize', range_start=self.dataset_size_start, range_end=self.dataset_size_end, range_steps=self.dataset_size_steps),
                                                VariableParam('synthratio', range_start=self.synth_ratio_start, range_end=self.synth_ratio_end, range_steps=self.synth_ratio_steps),
                                                VariableParamInt('model', range_start=0, range_end=self.classifiers_per_nf, range_steps=self.classifiers_per_nf),
                                                MetricParam('loss'),
                                                MetricParam('accuracy'),
                                                MetricParam('max_epoch'),
                                                MetricParam('tnoise'),
                                                MetricParam('fnoise'),
                                                MetricParam('tsig'),
                                                MetricParam('fsig'),
                                                CopyFromParam('clfsize', source_param='dsize'),
                                                LambdaParams.clf_t_g_size_from_clfsize_synthratio(val_size=self.val_size),
                                                LambdaParams.clf_t_s_size_from_clfsize_synthratio(val_size=self.val_size),
                                                LambdaParams.clf_v_g_size_from_clfsize_synthratio(val_size=self.val_size),
                                                LambdaParams.clf_v_s_size_from_clfsize_synthratio(val_size=self.val_size)) \
            .build_plan() \
            .label('synthratio', 'Synth Ratio') \
            .label('dsize', 'Samples seen by NF')
        p = self.training_planner.plan
        print('ALTA!')
