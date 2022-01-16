import math
from pathlib import Path

from typing import Type

from distributions.LearnedDistribution import LearnedDistributionCreator
from maf.DL import DL2
from maf.DS import DatasetProps
from maf.mixlearn.MixLearnExperimentMiniBoone import MixLearnExperimentMiniBoone
from maf.mixlearn.dl3.MinibooneDL3 import MinibooneDL3
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
from maf.variable.TrainingPlanner import TrainingPlanner
from maf.variable.VariableParam import FixedParam, LambdaParams, VariableParamInt, MetricParam, MetricIntParam, LambdaParam, FixedIntParam, LambdaIntParam


class MixLearnExperimentMiniBooneDSizeVar(MixLearnExperimentMiniBoone):
    def __init__(self, name: str,
                 learned_distr_creator: LearnedDistributionCreator,
                 dataset_name: str,
                 result_folder: Path,
                 epochs: int,
                 batch_size: int = 128,
                 paper_load: bool = False,
                 experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess,
                 test_split: float = 0.1,
                 classifiers_per_nf: int = 3,
                 just_signal_plan: bool = False):
        super().__init__(
            classifiers_per_nf=classifiers_per_nf,
            name=name,
            learned_distr_creator=learned_distr_creator,
            dataset_name=dataset_name,
            result_folder=result_folder,
            epochs=epochs,
            batch_size=batch_size,
            paper_load=paper_load,
            experiment_init_ds_class=experiment_init_ds_class,
            test_split=test_split)
        self.just_signal_plan: bool = just_signal_plan

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(self.cache_dir, "miniboone_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def create_data_loader(self, norm_data: bool) -> DL2:
        dl3 = MinibooneDL3(dataset_name=self.dataset_name)
        return dl3.execute()

    def _create_training_plan(self):
        props: DatasetProps = self.dl_training.props
        test_props: DatasetProps = self.dl_test.props
        signal_ratio: float = props.no_of_signals / (props.no_of_signals + props.no_of_noise)
        noise_ratio: float = props.no_of_noise / (props.no_of_signals + props.no_of_noise)

        TrainingPlanner(FixedParam('clf_t_g_size', 15000),
                        VariableParamInt('clfsize', range_start=16500, range_end=self.dataset_size_end, range_steps=10),
                        LambdaParams.clf_t_s_size_from_clf_t_g_size_clfsize(val_size=self.val_size),
                        LambdaParams.clf_v_g_size_from_clf_t_g_size_clf_t_s_size(val_size=self.val_size),
                        LambdaParams.clf_v_s_size_from_clf_v_g_size(val_size=self.val_size))
        # for CONDITIONAL or learning SIGNAL AND NOISE
        # return TrainingPlanner(FixedIntParam('done', 0),
        #                        LambdaParam('tsize', source_params=['dsize'], f=lambda dsize: dsize - self.val_size),
        #                        FixedParam('vsize', self.val_size),
        #                        VariableParamInt('dsize', range_start=16500, range_end=self.dataset_size_end, range_steps=3, is_var=True),  # 10
        #                        VariableParamInt('model', range_start=1, range_end=self.classifiers_per_nf, range_steps=self.classifiers_per_nf),
        #                        MetricParam('loss'),
        #                        MetricParam('accuracy'),
        #                        MetricIntParam('max_epoch'),
        #                        MetricIntParam('tnoise'),
        #                        MetricIntParam('fnoise'),
        #                        MetricIntParam('tsig'),
        #                        MetricIntParam('fsig'),
        #                        # LambdaParam('size_clf_t_ge', source_params=['clfsize'], f=lambda clfsize: clfsize - self.val_size),
        #                        FixedIntParam('size_clf_t_ge', 16500 - self.val_size),
        #                        LambdaIntParam('size_clf_t_sy', source_params=['size_clf_t_ge', 'clfsize'], f=lambda tge, clfsize: clfsize - self.val_size - tge),
        #                        LambdaIntParam('size_clf_v_ge', source_params=['size_clf_t_ge', 'size_clf_t_sy'], f=lambda tge, tsy: math.floor(tge / (tge + tsy) * self.val_size)),
        #                        LambdaIntParam('size_clf_v_sy', source_params=['size_clf_t_ge', 'size_clf_t_sy'], f=lambda tge, tsy: math.ceil(tsy / (tge + tsy) * self.val_size)),
        #
        #                        LambdaIntParam('clf_t_ge_sig', source_params=['size_clf_t_ge'], f=lambda tge: math.floor(signal_ratio * tge)),
        #                        LambdaIntParam('clf_t_ge_noi', source_params=['size_clf_t_ge'], f=lambda tge: math.ceil(noise_ratio * tge)),
        #                        LambdaIntParam('clf_t_sy_sig', source_params=['size_clf_t_sy'], f=lambda tsy: math.floor(signal_ratio * tsy)),
        #                        LambdaIntParam('clf_t_sy_noi', source_params=['size_clf_t_sy'], f=lambda tsy: math.ceil(noise_ratio * tsy)),
        #                        LambdaIntParam('debug', source_params=['clf_t_ge_sig', 'clf_t_sy_sig', 'clf_t_ge_noi', 'clf_t_sy_noi'],
        #                                       f=lambda a, b, c, d: (a + b + c + d + self.val_size)),
        #
        #                        LambdaIntParam('clf_v_ge_sig', source_params=['size_clf_v_ge'], f=lambda vge: math.floor(signal_ratio * vge)),
        #                        LambdaIntParam('clf_v_ge_noi', source_params=['size_clf_v_ge'], f=lambda vge: math.ceil(noise_ratio * vge)),
        #                        LambdaIntParam('clf_v_sy_sig', source_params=['size_clf_v_sy'], f=lambda vsy: math.floor(signal_ratio * vsy)),
        #                        LambdaIntParam('clf_v_sy_noi', source_params=['size_clf_v_sy'], f=lambda vsy: math.ceil(noise_ratio * vsy)),
        #
        #                        VariableParamInt('clfsize', range_start=16500, range_end=self.dataset_size_end, range_steps=3),  # 10
        #                        LambdaIntParam('size_nf_t_sig', source_params=['tsize'], f=lambda tsize: tsize),
        #                        FixedIntParam('size_nf_v_sig', 1500),  # Genuine Signal Val for MAF
        #                        FixedIntParam('size_nf_t_noi', 0),
        #                        FixedIntParam('size_nf_v_noi', 0),
        #                        FixedIntParam('test_clf_sig', test_props.no_of_signals),
        #                        FixedIntParam('test_clf_no', test_props.no_of_noise))
        if self.just_signal_plan:
            return TrainingPlanner(FixedIntParam('done', 0),
                                   LambdaIntParam('tsize', source_params=['dsize'], f=lambda dsize: dsize - self.val_size),  # todo obsolete
                                   VariableParamInt('dsize', range_start=16500, range_end=self.dataset_size_end, range_steps=10, is_var=True),  # 10
                                   VariableParamInt('model', range_start=1, range_end=self.classifiers_per_nf, range_steps=self.classifiers_per_nf),
                                   MetricParam('loss'),
                                   MetricParam('accuracy'),
                                   MetricIntParam('max_epoch'),
                                   MetricIntParam('tnoise'),
                                   MetricIntParam('fnoise'),
                                   MetricIntParam('tsig'),
                                   MetricIntParam('fsig'),
                                   FixedIntParam('size_clf_t_ge', 16500 - self.val_size),
                                   LambdaIntParam('size_clf_t_sy', source_params=['size_clf_t_ge', 'clfsize'], f=lambda tge, clfsize: clfsize - self.val_size - tge,
                                                  is_var=True),
                                   LambdaIntParam('size_clf_v_ge', source_params=['size_clf_t_ge', 'size_clf_t_sy'],
                                                  f=lambda tge, tsy: math.floor(tge / (tge + tsy) * self.val_size)),
                                   LambdaIntParam('size_clf_v_sy', source_params=['size_clf_t_ge', 'size_clf_t_sy'],
                                                  f=lambda tge, tsy: math.ceil(tsy / (tge + tsy) * self.val_size)),

                                   LambdaIntParam('clf_t_ge_sig', source_params=['size_clf_t_ge'], f=lambda tge: round(signal_ratio * tge)),
                                   LambdaIntParam('clf_t_ge_noi', source_params=['size_clf_t_ge', 'size_clf_t_sy'],
                                                  f=lambda size_clf_t_ge, size_clf_t_sy: round(
                                                      size_clf_t_ge - (size_clf_t_ge * signal_ratio) + size_clf_t_sy - (size_clf_t_sy * signal_ratio))),
                                   LambdaIntParam('clf_t_sy_sig', source_params=['size_clf_t_sy'], f=lambda tsy: round(signal_ratio * tsy)),
                                   FixedIntParam('clf_t_sy_noi', 0),
                                   LambdaIntParam('clf_v_ge_sig', source_params=['size_clf_v_ge'], f=lambda vge: round(signal_ratio * vge)),
                                   LambdaIntParam('clf_v_ge_noi', source_params=['size_clf_v_ge', 'clf_v_ge_sig', 'size_clf_v_sy'],
                                                  f=lambda vge, vgesig, vsy: round(vge - vgesig + noise_ratio * vsy)),
                                   LambdaIntParam('clf_v_sy_sig', source_params=['size_clf_v_sy'], f=lambda vsy: round(signal_ratio * vsy)),
                                   FixedIntParam('clf_v_sy_noi', 0),

                                   VariableParamInt('clfsize', range_start=16500, range_end=self.dataset_size_end, range_steps=10),  # 10
                                   LambdaIntParam('size_nf_t_sig', source_params=['dsize'], f=lambda dsize: math.floor(signal_ratio * dsize - self.val_size)),
                                   FixedIntParam('size_nf_v_sig', 1500),  # Genuine Signal Val for MAF
                                   FixedIntParam('size_nf_t_noi', 0),
                                   FixedIntParam('size_nf_v_noi', 0),
                                   FixedIntParam('test_clf_sig', test_props.no_of_signals),
                                   FixedIntParam('test_clf_no', test_props.no_of_noise))

        with_noise_plan = TrainingPlanner(FixedIntParam('done', 0),
                                          LambdaIntParam('tsize', source_params=['dsize'], f=lambda dsize: dsize - self.val_size),  # todo obsolete
                                          VariableParamInt('dsize', range_start=16500, range_end=self.dataset_size_end, range_steps=10, is_var=True),  # 10
                                          VariableParamInt('model', range_start=1, range_end=self.classifiers_per_nf, range_steps=self.classifiers_per_nf),
                                          MetricParam('loss'),
                                          MetricParam('accuracy'),
                                          MetricIntParam('max_epoch'),
                                          MetricIntParam('tnoise'),
                                          MetricIntParam('fnoise'),
                                          MetricIntParam('tsig'),
                                          MetricIntParam('fsig'),
                                          FixedIntParam('size_clf_t_ge', 16500 - self.val_size),
                                          LambdaIntParam('size_clf_t_sy', source_params=['size_clf_t_ge', 'clfsize'], f=lambda tge, clfsize: clfsize - self.val_size - tge,
                                                         is_var=True),
                                          LambdaIntParam('size_clf_v_ge', source_params=['size_clf_t_ge', 'size_clf_t_sy'],
                                                         f=lambda tge, tsy: math.floor(tge / (tge + tsy) * self.val_size)),
                                          LambdaIntParam('size_clf_v_sy', source_params=['size_clf_t_ge', 'size_clf_t_sy'],
                                                         f=lambda tge, tsy: math.ceil(tsy / (tge + tsy) * self.val_size)),

                                          LambdaIntParam('clf_t_ge_sig', source_params=['size_clf_t_ge'], f=lambda tge: round(signal_ratio * tge)),
                                          LambdaIntParam('clf_t_ge_noi', source_params=['size_clf_t_ge'], f=lambda tge: round(tge - (signal_ratio * tge))),
                                          LambdaIntParam('clf_t_sy_sig', source_params=['size_clf_t_sy'], f=lambda tsy: round(signal_ratio * tsy)),
                                          LambdaIntParam('clf_t_sy_noi', source_params=['size_clf_t_sy'], f=lambda tsy: round(tsy - (signal_ratio * tsy))),
                                          LambdaIntParam('debug', source_params=['clf_t_ge_sig', 'clf_t_sy_sig', 'clf_t_ge_noi', 'clf_t_sy_noi'],
                                                         f=lambda a, b, c, d: (a + b + c + d + self.val_size)),

                                          LambdaIntParam('clf_v_ge_sig', source_params=['size_clf_v_ge'], f=lambda vge: round(signal_ratio * vge)),
                                          LambdaIntParam('clf_v_ge_noi', source_params=['size_clf_v_ge'], f=lambda vge: round(vge - signal_ratio * vge)),
                                          LambdaIntParam('clf_v_sy_sig', source_params=['size_clf_v_sy'], f=lambda vsy: round(signal_ratio * vsy)),
                                          LambdaIntParam('clf_v_sy_noi', source_params=['size_clf_v_sy'], f=lambda vsy: round(vsy - signal_ratio * vsy)),

                                          VariableParamInt('clfsize', range_start=16500, range_end=self.dataset_size_end, range_steps=10),  # 10
                                          LambdaIntParam('size_nf_t_sig', source_params=['tsize'], f=lambda tsize: round(signal_ratio * (tsize))),
                                          FixedIntParam('size_nf_v_sig', round(signal_ratio * self.val_size)),  # Genuine Signal Val for MAF
                                          LambdaIntParam('size_nf_t_noi', source_params=['tsize'], f=lambda tsize: round(tsize - (tsize * signal_ratio))),
                                          FixedIntParam('size_nf_v_noi', round(self.val_size - (signal_ratio * self.val_size))),
                                          FixedIntParam('test_clf_sig', test_props.no_of_signals),
                                          FixedIntParam('test_clf_no', test_props.no_of_noise))
        return with_noise_plan
