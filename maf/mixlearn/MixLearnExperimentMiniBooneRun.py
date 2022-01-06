from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MAFCreator
from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone

if __name__ == '__main__':
    ArgParser.parse()
    creator = MAFCreator(batch_norm=True,
                         conditional_one_hot=True,
                         # epochs=epochs,
                         hidden_shape=[1024, 1024],
                         input_noise_variance=0.0,
                         layers=30,
                         norm_layer=False,
                         use_tanh_made=True)
    MixLearnExperimentMiniBoone.main_static(dataset_name='miniboone', experiment_name='miniboone_default', learned_distr_creator=creator)
