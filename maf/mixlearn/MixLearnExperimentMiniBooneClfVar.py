import math

from maf.mixlearn.dsinit.DSBalanceInitProcess import DSBalanceInitProcess
from pathlib import Path

from typing import Type

from distributions.LearnedDistribution import LearnedDistributionCreator
from maf.DL import DL2, DL3
from maf.DS import DatasetProps
from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone
from maf.mixlearn.dl3.MinibooneDL3 import MinibooneDL3
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
from maf.variable.TrainingPlanner import TrainingPlanner
from maf.variable.VariableParam import FixedParam, LambdaParams, VariableParamInt, MetricParam, MetricIntParam, LambdaParam, FixedIntParam, LambdaIntParam, VariableParam


class MixLearnExperimentMiniBooneClfVar(MixLearnExperimentMiniBoone):
    def __init__(self, name: str,
                 learned_distr_creator: LearnedDistributionCreator,
                 dataset_name: str,
                 result_folder: Path,
                 clf_epochs: int,
                 nf_epochs: int,
                 clf_patience: int = 10,
                 nf_patience: int = 10,
                 batch_size: int = 128,
                 paper_load: bool = False,
                 experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess,
                 test_split: float = 0.1,
                 classifiers_per_nf: int = 3,
                 pool_size: int = 6,
                 just_signal_plan: bool = False,
                 synth_samples_amount_multiplier: float = 1.0,
                 sample_variance_multiplier: float = 1.0,
                 steps_size_clf_t_ge: int = 10,
                 steps_size_clf_t_sy: int = 10):
        super().__init__(
            classifiers_per_nf=classifiers_per_nf,
            name=name,
            learned_distr_creator=learned_distr_creator,
            dataset_name=dataset_name,
            result_folder=result_folder,
            clf_epochs=clf_epochs,
            clf_patience=clf_patience,
            nf_epochs=nf_epochs,
            nf_patience=nf_patience,
            batch_size=batch_size,
            paper_load=paper_load,
            experiment_init_ds_class=experiment_init_ds_class,
            test_split=test_split,
            pool_size=pool_size,
            sample_variance_multiplier=sample_variance_multiplier)
        self.just_signal_plan: bool = just_signal_plan
        self.synth_samples_amount_multiplier: float = synth_samples_amount_multiplier
        self.steps_size_clf_t_ge: int = steps_size_clf_t_ge
        self.steps_size_clf_t_sy: int = steps_size_clf_t_sy

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(self.cache_dir, "miniboone_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def _create_training_plan(self):
        props: DatasetProps = self.dl_training.props
        test_props: DatasetProps = self.dl_test.props
        signal_ratio: float = props.no_of_signals / (props.no_of_signals + props.no_of_noise)
        noise_ratio: float = props.no_of_noise / (props.no_of_signals + props.no_of_noise)

        with_noise_plan = TrainingPlanner(FixedIntParam('done', 0),
                                          LambdaIntParam('tsize', source_params=['dsize'], f=lambda dsize: dsize - self.val_size),  # todo obsolete
                                          FixedIntParam('dsize', props.length),
                                          VariableParamInt('model', range_start=1, range_end=self.classifiers_per_nf, range_steps=self.classifiers_per_nf),
                                          MetricParam('loss'),
                                          MetricParam('accuracy'),
                                          MetricIntParam('max_epoch'),
                                          MetricIntParam('tnoise'),
                                          MetricIntParam('fnoise'),
                                          MetricIntParam('tsig'),
                                          MetricIntParam('fsig'),

                                          VariableParamInt('size_clf_t_ge', range_start=0, range_end=props.length - 1500, range_steps=self.steps_size_clf_t_ge, is_var=True),
                                          # FixedIntParam('size_clf_t_ge', 0, is_var=True),
                                          VariableParamInt('size_clf_t_sy', range_start=0, range_end=(props.length * self.synth_samples_amount_multiplier - 1500),
                                                           range_steps=self.steps_size_clf_t_sy,
                                                           is_var=True),
                                          LambdaIntParam('size_clf_v_ge', source_params=['size_clf_t_ge', 'size_clf_t_sy'],
                                                         f=lambda tge, tsy: math.floor(tge / (tge + tsy) * self.val_size) if tge > 0 or tsy > 0 else 0),
                                          LambdaIntParam('size_clf_v_sy', source_params=['size_clf_t_ge', 'size_clf_t_sy'],
                                                         f=lambda tge, tsy: math.ceil(tsy / (tge + tsy) * self.val_size) if tge > 0 or tsy > 0 else 0),

                                          LambdaIntParam('clf_t_ge_sig', source_params=['size_clf_t_ge'], f=lambda tge: round(signal_ratio * tge)),
                                          LambdaIntParam('clf_t_ge_noi', source_params=['size_clf_t_ge'], f=lambda tge: round(tge - (signal_ratio * tge))),
                                          LambdaIntParam('clf_t_sy_sig', source_params=['size_clf_t_sy'], f=lambda tsy: round(signal_ratio * tsy)),
                                          LambdaIntParam('clf_t_sy_noi', source_params=['size_clf_t_sy'], f=lambda tsy: round(tsy - (signal_ratio * tsy))),

                                          LambdaIntParam('clf_v_ge_sig', source_params=['size_clf_v_ge'], f=lambda vge: round(signal_ratio * vge)),
                                          LambdaIntParam('clf_v_ge_noi', source_params=['size_clf_v_ge'], f=lambda vge: round(vge - signal_ratio * vge)),
                                          LambdaIntParam('clf_v_sy_sig', source_params=['size_clf_v_sy'], f=lambda vsy: round(signal_ratio * vsy)),
                                          LambdaIntParam('clf_v_sy_noi', source_params=['size_clf_v_sy'], f=lambda vsy: round(vsy - signal_ratio * vsy)),

                                          LambdaIntParam('clfsize', source_params=['size_clf_t_ge', 'size_clf_t_sy', 'size_clf_v_ge', 'size_clf_v_sy'],
                                                         f=lambda tge, tsy, vge, vsy: tge + tsy + vge + vsy),

                                          LambdaIntParam('size_nf_t_sig', source_params=['tsize'], f=lambda tsize: round(signal_ratio * (tsize))),
                                          FixedIntParam('size_nf_v_sig', round(signal_ratio * self.val_size)),  # Genuine  Signal Val for MAF
                                          LambdaIntParam('size_nf_t_noi', source_params=['tsize'], f=lambda tsize: round(tsize - (tsize * signal_ratio))),
                                          FixedIntParam('size_nf_v_noi', round(self.val_size - (signal_ratio * self.val_size))),
                                          FixedIntParam('test_clf_sig', test_props.no_of_signals),
                                          FixedIntParam('test_clf_no', test_props.no_of_noise)) \
            .label('size_clf_t_ge', 'Genuine samples seen by classifier') \
            .label('size_clf_t_sy', 'Synthetic samples seen by classifier')
        return with_noise_plan
