from distributions.LearnedGaussianMultivariate import LearnedGaussianMultivariateCreatorCov
from keta.argparseer import ArgParser
from maf.MaskedAutoregressiveFlow import MAFCreator
from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone


if __name__ == '__main__':
    ArgParser.parse()
    creator = LearnedGaussianMultivariateCreatorCov()
    MixLearnExperimentMiniBoone.main_static(dataset_name='miniboone', experiment_name='miniboone_MVG2', learned_distr_creator=creator)
