from maf.MaskedAutoregressiveFlow import MAFCreator
from maf.mixlearn.MixLearnExperimentSynth import MixLeanExperimentSynth

if __name__ == '__main__':
    creator = MAFCreator(batch_norm=True,
                         norm_layer=False,
                         use_tanh_made=True,
                         input_noise_variance=0.0,
                         layers=5,
                         conditional_one_hot=False,
                         hidden_shape=[200, 200])
    MixLeanExperimentSynth.run_me(creator=creator,
                                  dim=5,
                                  experiment_name='ex_synth1_5d-norm_new',
                                  dataset_name='ds_synth1_5d_norm')
