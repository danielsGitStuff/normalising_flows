from keta.argparseer import ArgParser

from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone
from maf.mixlearn.dsinit.DSOutlierInitProcess import DSOutlierInitProcess

if __name__ == '__main__':
    ArgParser.parse()
    MixLearnExperimentMiniBoone.main_static(dataset_name='miniboone_no_outliers',
                                            experiment_name='miniboone_no_outliers_var_clf_t_size',
                                            experiment_init_ds_class=DSOutlierInitProcess)
