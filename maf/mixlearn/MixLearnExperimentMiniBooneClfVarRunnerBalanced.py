from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner import MixLearnExperimentMiniBooneClfVarRunner
from maf.mixlearn.dsinit.DSBalanceInitProcess import DSBalanceInitProcess


class MixLearnExperimentMiniBooneClfVarRunnerBalanced(MixLearnExperimentMiniBooneClfVarRunner):
    def __init__(self):
        super().__init__(name='miniboone_clf_var_balanced')
        self.experiment_init_ds_class = DSBalanceInitProcess