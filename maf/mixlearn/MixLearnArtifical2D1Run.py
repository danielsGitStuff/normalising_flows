from pathlib import Path

from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MAFCreator
from maf.mixlearn.MixLearnArtifical2D1 import MixLearnArtificial2D1

if __name__ == '__main__':
    ArgParser.parse()
    creator = MAFCreator(batch_norm=True,
                         conditional_one_hot=False,
                         # epochs=epochs,
                         hidden_shape=[200, 200],
                         input_noise_variance=0.0,
                         layers=4,
                         norm_layer=False,
                         use_tanh_made=True)
    results_folder = Path('results_artificial')
    experiment = MixLearnArtificial2D1(name='artificial2d1',
                                                     learned_distribution_creator=creator,
                                                     result_folder=results_folder,
                                                     epochs=100,
                                                     batch_size=1000,
                                                     classifiers_per_nf=1,
                                                     just_signal_plan=False)
    experiment.create_training_plan().run()
