from keta.argparseer import ArgParser
from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner import MixLearnExperimentMiniBooneClfVarRunner
from maf.mixlearn.dsinit.DSBalanceInitProcess import DSBalanceInitProcess


class MixLearnExperimentMiniBooneClfVarRunnerBalancedBig(MixLearnExperimentMiniBooneClfVarRunner):
    def __init__(self):
        super().__init__(name='miniboone_clf_var_balanced', pool_size=8, synth_samples_amount_multiplier=4.0)
        self.experiment_init_ds_class = DSBalanceInitProcess


if __name__ == '__main__':
    ArgParser.parse()
    # Global.Testing.set('testing_nf_layers', 1)
    # Global.Testing.set('testing_epochs', 1)
    MixLearnExperimentMiniBooneClfVarRunnerBalancedBig().run()
