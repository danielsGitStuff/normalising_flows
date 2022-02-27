from distributions.base import enable_memory_growth
from typing import Type

from common.globals import Global
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
from maf.stuff.MafExperiment import MafExperiment
from pathlib import Path

from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MAFCreator
from maf.mixlearn.MixLearnExperimentMiniBooneClfVar import MixLearnExperimentMiniBooneClfVar


class MixLearnExperimentMiniBooneClfVarRunner(MafExperiment):
    def print_divergences(self):
        pass

    def _run(self):
        # creator = MAFCreator(batch_norm=True,
        #                      conditional_one_hot=False,
        #                      # epochs=epochs,
        #                      hidden_shape=[100, 100],
        #                      input_noise_variance=0.0,
        #                      layers=20,
        #                      norm_layer=False,
        #                      use_tanh_made=True)
        # results_folder = Global.get_default('results_dir', Path('results_miniboone'))
        # experiment = MixLearnExperimentMiniBooneClfVar(name=self.name,
        #                                                learned_distr_creator=creator,
        #                                                dataset_name='miniboone',
        #                                                result_folder=results_folder,
        #                                                experiment_init_ds_class=self.experiment_init_ds_class,
        #                                                paper_load=False,
        #                                                clf_epochs=1000,
        #                                                clf_patience=20,
        #                                                nf_epochs=2000,
        #                                                nf_patience=20,
        #                                                batch_size=1024,
        #                                                classifiers_per_nf=1,
        #                                                just_signal_plan=False)
        #
        creator = MAFCreator(batch_norm=False,
                             conditional_one_hot=False,
                             # epochs=epochs,
                             hidden_shape=[200, 200],
                             input_noise_variance=0.0,
                             layers=30,
                             norm_layer=False,
                             use_tanh_made=True)
        results_folder = Global.get_default('results_dir', Path('results_miniboone'))
        experiment = MixLearnExperimentMiniBooneClfVar(name=self.name,
                                                       learned_distr_creator=creator,
                                                       dataset_name='miniboone',
                                                       result_folder=results_folder,
                                                       experiment_init_ds_class=self.experiment_init_ds_class,
                                                       paper_load=False,
                                                       clf_epochs=1000,
                                                       clf_patience=20,
                                                       nf_epochs=2000,
                                                       nf_patience=20,
                                                       batch_size=1024,
                                                       classifiers_per_nf=3,
                                                       just_signal_plan=False,
                                                       pool_size=self.pool_size,
                                                       synth_samples_amount_multiplier=self.synth_samples_amount_multiplier,
                                                       steps_size_clf_t_ge=self.steps_size_clf_t_ge,
                                                       steps_size_clf_t_sy=self.steps_size_clf_t_sy,
                                                       sample_variance_multiplier=self.sample_variance_multiplier)

        experiment.create_training_plan().run()

    def __init__(self, name='miniboone_clfvar', pool_size: int = 8, synth_samples_amount_multiplier: float = 1.0, steps_size_clf_t_ge: int = 10, steps_size_clf_t_sy: int = 10,
                 sample_variance_multiplier: float = 1.0):
        super().__init__(name, pool_size=pool_size)
        self.synth_samples_amount_multiplier: float = synth_samples_amount_multiplier
        self.experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess
        self.steps_size_clf_t_ge: int = steps_size_clf_t_ge
        self.steps_size_clf_t_sy: int = steps_size_clf_t_sy
        self.sample_variance_multiplier: float = sample_variance_multiplier


if __name__ == '__main__':
    ArgParser.parse()
    # Global.Testing.set('testing_nf_layers', 1)
    # Global.Testing.set('testing_epochs', 1)
    MixLearnExperimentMiniBooneClfVarRunner().run()
