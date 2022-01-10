from maf.mixlearn.MixLearnExperimentMiniBooneDSizeVar import MixLearnExperimentMiniBooneDSizeVar
from maf.mixlearn.MixLearnExperimentMiniBooneSynthVar import MixLearnExperimentMiniBooneSynthVar
from pathlib import Path

from common.globals import Global
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MAFCreator
from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone

if __name__ == '__main__':
    ArgParser.parse()
    Global.set_global('testing_noise', True)
    creator = MAFCreator(batch_norm=True,
                         conditional_one_hot=True,
                         # epochs=epochs,
                         hidden_shape=[1024, 1024],
                         input_noise_variance=0.0,
                         layers=30,
                         norm_layer=False,
                         use_tanh_made=True)
    results_folder = Path('results_miniboone')
    experiment = MixLearnExperimentMiniBooneDSizeVar(name='miniboone_default',
                                learned_distr_creator=creator,
                                dataset_name='miniboone',
                                result_folder=results_folder,
                                paper_load=False,
                                epochs=100,
                                batch_size=1000)
    experiment.create_training_plan().run()
