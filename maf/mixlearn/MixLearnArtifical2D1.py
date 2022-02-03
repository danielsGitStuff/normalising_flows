import sys
from pathlib import Path

import math
from typing import Type

from common import jsonloader
from common.globals import Global
from common.jsonloader import Ser
from distributions.LearnedDistribution import LearnedDistributionCreator
from maf.DL import DL2, DL3
from maf.DS import DatasetProps
from maf.mixlearn.ClassifierTrainingProcess import BinaryClassifierCreator
from maf.mixlearn.MixLearnExperiment import MixLearnExperiment
from maf.mixlearn.classifiers.Arttificial2D1 import Artificial2D1ClassifierCreator
from maf.mixlearn.dl3.ArtificialDL3 import ArtificialIntersection2DDL3, ArtificialDL3
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
from maf.variable.TrainingPlanner import TrainingPlanner
from maf.variable.VariableParam import VariableParamInt, MetricParam, MetricIntParam, FixedIntParam, LambdaIntParam


class ArtificialPrinter(Ser):
    @staticmethod
    def static_execute(js: str):
        p: ArtificialPrinter = jsonloader.from_json(js)
        p._run()

    def execute(self):
        js = jsonloader.to_json(self)
        Global.POOL().run_blocking(ArtificialPrinter.static_execute, js)

    def _run(self):
        print('running')

class ArtificalMixLearnExperiment(MixLearnExperiment):
    def __init__(self, name: str, learned_distribution_creator: LearnedDistributionCreator, classifier_creator: BinaryClassifierCreator, result_folder: Path, epochs: int,
                 classifiers_per_nf: int, batch_size: int):
        super().__init__(name,
                         learned_distribution_creator=learned_distribution_creator,
                         classifier_creator=classifier_creator,
                         result_folder=result_folder,
                         clf_epochs=epochs,
                         batch_size=batch_size)
        self.classifiers_per_nf = classifiers_per_nf  # todo move to super class

    def _run(self):
        self.start()

    def print_divergences(self):
        pass

    def start(self):
        # print dataset here if possible
        self._print_dataset()
        sys.exit(6)
        super(ArtificalMixLearnExperiment, self).start()




class MixLearnArtificial2D1(ArtificalMixLearnExperiment):
    def __init__(self, name: str,
                 learned_distribution_creator: LearnedDistributionCreator,
                 result_folder: Path,
                 epochs: int,
                 batch_size: int = 128,
                 experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess,
                 test_split: float = 0.1,
                 classifiers_per_nf: int = 3,
                 just_signal_plan: bool = False):
        super().__init__(
            classifier_creator=Artificial2D1ClassifierCreator(),
            learned_distribution_creator=learned_distribution_creator,
            classifiers_per_nf=classifiers_per_nf,
            name=name,
            result_folder=result_folder,
            epochs=epochs,
            batch_size=batch_size)
        self.just_signal_plan: bool = just_signal_plan

    def create_checkpoint_dir(self) -> Path:
        checkpoints_dir = Path(self.cache_dir, f"{self.name}_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    def create_dl3(self) -> DL3:
        return ArtificialIntersection2DDL3()

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

                                          VariableParamInt('size_clf_t_ge', range_start=0, range_end=props.length - 1500, range_steps=2, is_var=True),
                                          VariableParamInt('size_clf_t_sy', range_start=0, range_end=2 * props.length, range_steps=2, is_var=True),
                                          # VariableParam('clf_ge_sy_ratio', range_start=0.0, range_end=1.0, range_steps=3),
                                          # LambdaIntParam('size_clf_t_ge', source_params=['clf_ge_sy_ratio', 'clfsize'], f=lambda r, clfsize: round(r * (clfsize - 1500))),
                                          # LambdaIntParam('size_clf_t_sy', source_params=['clf_ge_sy_ratio', 'clfsize'], f=lambda r, clfsize: round((1 - r) * (clfsize - 1500))),

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

                                          # VariableParamInt('clfsize', range_start=2500, range_end=self.dataset_size_end, range_steps=3),  # 10
                                          LambdaIntParam('clfsize', source_params=['size_clf_t_ge', 'size_clf_t_sy', 'size_clf_v_ge', 'size_clf_v_sy'],
                                                         f=lambda tge, tsy, vge, vsy: tge + tsy + vge + vsy),

                                          LambdaIntParam('size_nf_t_sig', source_params=['tsize'], f=lambda tsize: round(signal_ratio * (tsize))),
                                          FixedIntParam('size_nf_v_sig', round(signal_ratio * self.val_size)),  # Genuine  Signal Val for MAF
                                          LambdaIntParam('size_nf_t_noi', source_params=['tsize'], f=lambda tsize: round(tsize - (tsize * signal_ratio))),
                                          FixedIntParam('size_nf_v_noi', round(self.val_size - (signal_ratio * self.val_size))),
                                          FixedIntParam('test_clf_sig', test_props.no_of_signals),
                                          FixedIntParam('test_clf_no', test_props.no_of_noise))
        return with_noise_plan