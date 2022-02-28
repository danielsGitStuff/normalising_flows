from __future__ import annotations
from pathlib import Path
from typing import Type

from maf.DL import DL2, DL3
from distributions.LearnedDistribution import LearnedDistributionCreator
from maf.mixlearn.classifiers.Miniboone import MiniBooneBinaryClassifierCreator
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
from maf.mixlearn.dl3.MinibooneDL3 import MinibooneDL3
from maf.mixlearn.MixLearnExperiment import MixLearnExperiment


class MixLearnExperimentMiniBoone(MixLearnExperiment):
    def __init__(self, name: str,
                 learned_distr_creator: LearnedDistributionCreator,
                 dataset_name: str,
                 result_folder: Path,
                 clf_epochs: int,
                 nf_epochs: int,
                 clf_patience: int = 10,
                 nf_patience: int = 10,
                 batch_size: int = 128,
                 paper_load: bool = False,
                 experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess,
                 test_split: float = 0.1,
                 classifiers_per_nf: int = 3,
                 pool_size: int = 6,
                 sample_variance_multiplier: float = 1.0):
        self.paper_load: bool = paper_load
        self.dataset_name: str = dataset_name

        super().__init__(name=name,
                         clf_epochs=clf_epochs,
                         clf_patience=clf_patience,
                         nf_epochs=nf_epochs,
                         nf_patience=nf_patience,
                         classifiers_per_nf=classifiers_per_nf,
                         # layers=layers,
                         batch_size=batch_size,
                         # hidden_shape=hidden_shape,c
                         # norm_layer=norm_layer,
                         # noise_variance=noise_variance,
                         # batch_norm=batch_norm,
                         # use_tanh_made=use_tanh_made,
                         sample_variance_multiplier=sample_variance_multiplier,
                         test_split=test_split,
                         result_folder=result_folder,
                         experiment_init_ds_class=experiment_init_ds_class,
                         learned_distribution_creator=learned_distr_creator,
                         classifier_creator=MiniBooneBinaryClassifierCreator(),
                         pool_size=pool_size)

    def print_divergences(self):
        pass

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(self.cache_dir, "miniboone_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def create_dl3(self) -> DL3:
        return MinibooneDL3(dataset_name=self.dataset_name)

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
                                        clf_epochs=100,
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
