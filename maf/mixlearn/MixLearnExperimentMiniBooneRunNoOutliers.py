import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from distributions.base import cast_dataset_to_tensor, cast_to_ndarray
from maf.DS import DS
from maf.DL import DL2, DataSource
from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone
from maf.mixlearn.dsinit.DSOutlierInitProcess import DSOutlierInitProcess

if __name__ == '__main__':
    MixLearnExperimentMiniBoone.main_static(dataset_name='miniboone_no_outliers',
                                            experiment_name='miniboone_no_outliers_var_clf_t_size',
                                            experiment_init_ds_class=DSOutlierInitProcess)
