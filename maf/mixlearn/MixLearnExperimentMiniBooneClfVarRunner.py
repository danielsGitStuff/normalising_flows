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
        creator = MAFCreator(batch_norm=True,
                             conditional_one_hot=False,
                             # epochs=epochs,
                             hidden_shape=[1024, 1024],
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
                                                       nf_epochs=4000,
                                                       nf_patience=100,
                                                       batch_size=1024,
                                                       classifiers_per_nf=1,
                                                       just_signal_plan=False)
        experiment.create_training_plan().run()

    def __init__(self, name='miniboone_clfvar'):
        super().__init__(name)
        self.experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess


if __name__ == '__main__':
    ArgParser.parse()
    # Global.Testing.set('testing_nf_layers', 1)
    # Global.Testing.set('testing_epochs', 1)
    MixLearnExperimentMiniBooneClfVarRunner().run()
