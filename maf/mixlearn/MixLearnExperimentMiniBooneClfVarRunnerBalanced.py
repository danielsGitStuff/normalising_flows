from keta.argparseer import ArgParser
from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner import MixLearnExperimentMiniBooneClfVarRunner
from maf.mixlearn.dsinit.DSBalanceInitProcess import DSBalanceInitProcess


class MixLearnExperimentMiniBooneClfVarRunnerBalanced(MixLearnExperimentMiniBooneClfVarRunner):
    def __init__(self):
        super().__init__(name='miniboone_clf_var_balanced',pool_size=2)
        self.experiment_init_ds_class = DSBalanceInitProcess


if __name__ == '__main__':
    ArgParser.parse()
    # Global.Testing.set('testing_nf_layers', 1)
    # Global.Testing.set('testing_epochs', 1)
    MixLearnExperimentMiniBooneClfVarRunnerBalanced().run()