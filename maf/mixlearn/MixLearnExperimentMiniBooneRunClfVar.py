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
        results_folder = Path('results_miniboone')
        experiment = MixLearnExperimentMiniBooneClfVar(name=self.name,
                                                       learned_distr_creator=creator,
                                                       dataset_name='miniboone',
                                                       result_folder=results_folder,
                                                       paper_load=False,
                                                       epochs=100,
                                                       batch_size=1000,
                                                       classifiers_per_nf=3,
                                                       just_signal_plan=False)
        experiment.create_training_plan().run()

    def __init__(self):
        super().__init__('miniboone_clfvar')


if __name__ == '__main__':
    ArgParser.parse()
    MixLearnExperimentMiniBooneClfVarRunner().run()
