from distributions.LearnedGaussianMultivariate import LearnedGaussianMultivariateCreatorVar
from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone

if __name__ == '__main__':
    creator = LearnedGaussianMultivariateCreatorVar()
    MixLearnExperimentMiniBoone.main_static(dataset_name='miniboone', experiment_name='miniboone_MVGvar', learned_distr_creator=creator)
