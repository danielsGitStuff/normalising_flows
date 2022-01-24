from __future__ import annotations

import functools
import gzip
import sys

import math
import os
import shutil
from keta.argparseer import ArgParser
from pathlib import Path
from typing import Optional, List, Dict, Type

import pandas as pd
import requests

from common import jsonloader
from common.NotProvided import NotProvided
from maf.DS import DatasetProps
from maf.DL import DL2
from maf.examples.stuff.MafExperiment import MafExperiment
from maf.examples.stuff.StaticMethods import StaticMethods
from maf.mixlearn.ClassifierTrainingProcess import ClassifierTrainingProcess, BinaryClassifierCreator
from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess
from maf.mixlearn.MAFTrainingProcess import MAFTrainingProcess
from distributions.LearnedDistribution import LearnedDistributionCreator
from maf.variable.TrainingPlanner import TrainingPlanner


class DatasetFetcher:
    def __init__(self, dataset_url: str, target_file: Path, extract: Optional[str] = None):
        self.dataset_url: str = dataset_url
        self.target_file: Path = target_file
        self.extract: Optional[str] = extract
        assert self.extract in {'gz', None}

    def fetch(self):

        def download(url: str, f: Path, binary: bool = False):
            print(f" downloading dataset '{url}' -> '{f}'")
            with requests.get(url, stream=True) as r:
                r.raw.read = functools.partial(r.raw.read, decode_content=True)
                with open(f, mode='wb') as out:
                    shutil.copyfileobj(r.raw, out, length=16 * 1024 * 1024)
            print(f"downloaded '{url}' -> '{f}'")

        if self.target_file.exists():
            return

        if self.extract is None:
            download(url=self.dataset_url, f=self.target_file)
        else:
            tmp: Path = self.target_file.with_name(f"{self.target_file.name}.download.tmp")
            download(url=self.dataset_url, f=tmp)
            with gzip.open(tmp) as gz:
                with open(self.target_file, mode='wb') as f:
                    shutil.copyfileobj(gz, f, length=16 * 1024 * 1024)
            os.remove(tmp)


class MixLearnExperiment(MafExperiment):

    def _run(self):
        pass

    def __init__(self, name: str,
                 learned_distribution_creator: LearnedDistributionCreator,
                 classifier_creator: BinaryClassifierCreator,
                 result_folder: Path,
                 epochs: int,
                 # layers: int,
                 batch_size: Optional[int] = None,
                 data_batch_size: Optional[int] = None,
                 # hidden_shape: List[int] = [200, 200],
                 # norm_layer: bool = True,
                 norm_data: bool = False,
                 # noise_variance: float = 0.0,
                 # batch_norm: bool = False,
                 # use_tanh_made: bool = False,
                 conditional: bool = False,
                 conditional_one_hot: bool = True,
                 load_limit: Optional[int] = None,
                 val_split: float = 0.1,
                 test_split: float = 0.1,
                 sample_variance_multiplier: float = 1.0,
                 classifiers_per_nf: int = 3,
                 experiment_init_ds_class: Type[DSInitProcess] = DSInitProcess):
        super().__init__(name)
        self.learned_distribution_creator: LearnedDistributionCreator = learned_distribution_creator
        self.experiment_init_ds_class: Type[DSInitProcess] = experiment_init_ds_class
        self.sample_variance_multiplier: float = sample_variance_multiplier
        self.result_folder: Path = result_folder
        self.result_folder.mkdir(exist_ok=True)
        self.cache_dir = Path(StaticMethods.cache_dir(), f"mixlearn_{name}")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_training_plan_file: Path = Path(self.cache_dir, 'training.plan.csv')
        self.dl_test_folder: Path = Path(self.cache_dir, 'dl_test')
        self.dl_training_folder: Path = Path(self.cache_dir, 'dl_train')
        self.data_batch_size: Optional[int] = data_batch_size
        self.conditional_one_hot: bool = conditional_one_hot
        self.batch_size: Optional[int] = batch_size
        # self.use_tanh_made: bool = use_tanh_made
        # self.batch_norm: bool = batch_norm
        self.test_split: float = test_split
        # self.noise_variance: float = noise_variance
        # self.hidden_shape: List[int] = hidden_shape
        # self.layers: int = layers
        self.epochs: int = epochs
        # self.norm_layer: bool = norm_layer
        self.norm_data: bool = norm_data
        self.initial_dl2: DL2 = self.create_data_loader(norm_data)
        self.initial_dl2 = self.initial_dl2.create_data_in_process(normalise=norm_data)
        self.init_process: DSInitProcess = self.experiment_init_ds_class(dl_cache_dir=self.initial_dl2.dir, experiment_cache_dir=self.cache_dir, test_split=self.test_split)
        # self.init_process: DSInitProcess = self.create_ds_init_process(dl_cache_dir=self.initial_dl2.dir, experiment_cache_dir=self.cache_dir, test_split=self.test_split)
        DSInitProcess.execute(self.init_process)
        # self.dl_training_props: DatasetProps = DL2.load_props(self.dl_training_folder)
        self.dl_training: DL2 = DL2.load(self.dl_training_folder)
        self.dl_test: DL2 = DL2.load(self.dl_test_folder)
        self.conditional: bool = conditional
        self.conditional_dims: int = self.dl_training.conditional_dims
        self.checkpoint_dir: Path = self.create_checkpoint_dir()
        self.checkpoint_dir_noise: Path = self.checkpoint_dir.with_name(f"{self.checkpoint_dir.name}_noise")
        self.prefix: str = self.maf_prefix()
        self.val_split: float = val_split
        # account for the test split that comes later! (1 - test_split)
        self.load_limit: int = self.dl_training.props.length if load_limit is None else load_limit
        # self.load_limit: int = self.data_props.length if load_limit is None else load_limit

        self.no_of_synthetic_samples_per_batch: int = 1000
        self.dataset_size_start: int = 2500
        self.dataset_size_end: int = self.load_limit
        self.classifiers_per_nf: int = classifiers_per_nf
        self.val_size: int = 1500

        self.training_planner: Optional[TrainingPlanner] = None
        self.result_training_plan: Path = Path(self.result_folder, f"{self.name}_training_plan.png")
        self.result_confusion_matrices: Path = Path(self.result_folder, f"{self.name}_confusion.png")

    # def get_nf_file(self, dataset_size: int, extension: Optional[str] = None) -> Path:
    #     """@return the base file name for everything that just depends on a NF: sample file (.npy), nf file (.json)"""
    #     name = f"nf_{dataset_size}"
    #     if extension is not None:
    #         name = f"{name}.{extension}"
    #     return Path(self.cache_dir, name)

    def get_nf_file(self, size_nf_t_noi: int, size_nf_t_sig: int, size_nf_v_sig: int, size_nf_v_noi: int, extension: Optional[str] = None) -> Path:
        name = f"nf_ts{size_nf_t_sig}_tn{size_nf_t_noi}_vs{size_nf_v_sig}_vn_{size_nf_v_noi}"
        if extension is not None:
            name = f"{name}.{extension}"
        return Path(self.cache_dir, name)

    def get_classifier_name(self, training_size: int, synth_ratio: float, model_id: int, extension: Optional[str] = None) -> Path:
        """@return the base file name for everything that just depends on a classifier (witch itself depends on a NF): training history, model file"""
        name = f"cl_ts{training_size}_sr{synth_ratio}m{model_id}"
        if extension is not None:
            name = f"{name}.{extension}"
        return Path(self.cache_dir, name)

    def create_training_plan(self) -> MixLearnExperiment:
        self.training_planner = self._create_training_plan().build_plan()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):  # more options can be specified also
            print(self.training_planner.plan)
        # sys.exit(5)
        return self

    def _create_training_plan(self) -> TrainingPlanner:
        raise NotImplementedError()

    def start(self):
        self.__create_nfs__()
        self.__create_classifiers__()
        self.__print_training_plan__()

    def __create_nfs__(self):
        if self.training_planner is None:
            raise RuntimeError("Configure training parameters and then create a training plan by calling create_training_plan() first.")
        p: TrainingPlanner = self.training_planner
        plan = p.plan
        unique_nfs = plan[['size_nf_t_noi', 'size_nf_t_sig', 'size_nf_v_noi', 'size_nf_v_sig']].drop_duplicates()
        dataset_sizes = p.plan['dsize'].unique()
        for i in range(len(unique_nfs)):
            uniques = unique_nfs.iloc[i]
            size_nf_t_noi = int(uniques['size_nf_t_noi'])
            size_nf_t_sig = int(uniques['size_nf_t_sig'])
            size_nf_v_noi = int(uniques['size_nf_v_noi'])
            size_nf_v_sig = int(uniques['size_nf_v_sig'])

            related: pd.DataFrame = plan.loc[(plan['size_nf_t_noi'] == size_nf_t_noi) &
                                             (plan['size_nf_t_sig'] == size_nf_t_sig) &
                                             (plan['size_nf_v_noi'] == size_nf_v_noi) &
                                             (plan['size_nf_v_sig'] == size_nf_v_sig)]

            train_dir = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi, size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi, extension='genuine')
            val_dir = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi, size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi, extension='genuine.val')
            synth_dir = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi, size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi, extension='synth')
            synth_val_dir = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi, size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi,
                                             extension='synth.val')
            if synth_dir.exists() and synth_val_dir.exists():
                print(f"sample folder '{synth_dir}' already exists. skipping...")
                continue
            sample_sig_size: int = int(related['clf_t_sy_sig'].max())
            sample_noi_size: int = int(related['clf_t_sy_noi'].max())
            sample_sig_size_val: int = int(related['clf_v_sy_sig'].max())
            sample_noi_size_val: int = int(related['clf_v_sy_noi'].max())

            take_t_sig: int = int(max(related['clf_t_ge_sig'].max(), size_nf_t_sig))
            take_t_noi: int = int(max(related['clf_t_ge_noi'].max(), size_nf_t_noi))
            take_v_sig: int = int(max(related['clf_v_ge_sig'].max(), size_nf_v_sig))
            take_v_noi: int = int(max(related['clf_v_ge_noi'].max(), size_nf_v_noi))

            # take_t_sig += take_v_sig
            # take_t_noi += take_v_noi

            nf_bas_file_name: str = self.get_nf_file(size_nf_t_noi=size_nf_t_noi,
                                                     size_nf_t_sig=size_nf_t_sig,
                                                     size_nf_v_noi=size_nf_v_noi,
                                                     size_nf_v_sig=size_nf_v_sig).name

            mtp = MAFTrainingProcess(train_dir=train_dir,
                                     learned_distribution_creator=self.learned_distribution_creator,
                                     val_dir=val_dir,
                                     synth_dir=synth_dir,
                                     synth_val_dir=synth_val_dir,
                                     epochs=10,  # self.epochs,
                                     cache_dir=self.cache_dir,
                                     checkpoint_dir_noise=self.checkpoint_dir_noise,
                                     dl_init=self.initial_dl2,
                                     batch_size=self.batch_size,

                                     size_nf_t_noi=size_nf_t_noi,
                                     size_nf_t_sig=size_nf_t_sig,
                                     size_nf_v_noi=size_nf_v_noi,
                                     size_nf_v_sig=size_nf_v_sig,

                                     gen_sig_samples=sample_sig_size,
                                     gen_noi_samples=sample_noi_size,
                                     gen_val_sig_samples=sample_sig_size_val,
                                     gen_val_noi_samples=sample_noi_size_val,

                                     take_t_sig=take_t_sig,
                                     take_t_noi=take_t_noi,
                                     take_v_sig=take_v_sig,
                                     take_v_noi=take_v_noi,
                                     checkpoint_dir=self.checkpoint_dir,
                                     nf_base_file_name=nf_bas_file_name,
                                     conditional=self.conditional,
                                     conditional_classes=self.dl_training.props.classes,
                                     conditional_one_hot=self.conditional_one_hot,
                                     # val_size=val_size,
                                     sample_variance_multiplier=self.sample_variance_multiplier
                                     )
            mtp.execute()
        return

    def __create_classifiers__(self):
        if self.cache_training_plan_file.exists():
            plan: pd.DataFrame = pd.read_csv(self.cache_training_plan_file, index_col=False)
        else:
            plan: pd.DataFrame = self.training_planner.plan
        todo = plan.loc[plan['done'] < 1.0]
        ds_test_folder = Path(self.cache_dir, 'dl_test')
        for index, row in todo.iterrows():
            print(f"training {index + 1}/{len(todo)} classifiers")
            dataset_size = int(row['dsize'])

            history_csv_file = Path(self.cache_dir, self.training_planner.get_classifier_name(row=row, extension='csv'))
            model_base_file = Path(self.cache_dir, self.training_planner.get_classifier_name(row=row))

            size_nf_t_noi = int(row['size_nf_t_noi'])
            size_nf_t_sig = int(row['size_nf_t_sig'])
            size_nf_v_noi = int(row['size_nf_v_noi'])
            size_nf_v_sig = int(row['size_nf_v_sig'])
            ds_training_folder = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi,
                                                  size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi, extension='genuine')
            ds_val_folder = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi,
                                             size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi, extension='genuine.val')
            ds_synth_training_folder = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi,
                                                        size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi, extension='synth')
            ds_synth_val_folder = self.get_nf_file(size_nf_t_sig=size_nf_t_sig, size_nf_t_noi=size_nf_t_noi,
                                                   size_nf_v_sig=size_nf_v_sig, size_nf_v_noi=size_nf_v_noi, extension='synth.val')

            # ds_training_folder = self.get_nf_file(dataset_size, extension='genuine')
            # ds_val_folder = self.get_nf_file(dataset_size, extension='genuine.val')
            # ds_synth_training_folder = self.get_nf_file(dataset_size, extension='synth')
            # ds_synth_val_folder = self.get_nf_file(dataset_size, extension='synth.val')

            dl_training_genuine: DL2 = jsonloader.load_json(Path(ds_training_folder, 'dl2.json'), raise_on_404=True)
            dl_training_synth: DL2 = jsonloader.load_json(Path(ds_synth_training_folder, 'dl2.json'), raise_on_404=True)
            dl_val_genuine: DL2 = jsonloader.load_json(Path(ds_val_folder, 'dl2.json'), raise_on_404=True)
            dl_val_synth: DL2 = jsonloader.load_json(Path(ds_synth_val_folder, 'dl2.json'), raise_on_404=True)
            dl_test: DL2 = jsonloader.load_json(Path(ds_test_folder, 'dl2.json'), raise_on_404=True)
            cp = ClassifierTrainingProcess(dl_training_genuine=dl_training_genuine,
                                           dl_training_synth=dl_training_synth,
                                           dl_val_genuine=dl_val_genuine,
                                           dl_val_synth=dl_val_synth,
                                           dl_test=dl_test,
                                           epochs=self.epochs,
                                           history_csv_file=history_csv_file,
                                           conditional_dims=self.conditional_dims,
                                           batch_size=self.batch_size,
                                           clf_t_ge_sig=int(row['clf_t_ge_sig']),
                                           clf_t_ge_noi=int(row['clf_t_ge_noi']),
                                           clf_t_sy_sig=int(row['clf_t_sy_sig']),
                                           clf_t_sy_noi=int(row['clf_t_sy_noi']),

                                           clf_v_ge_sig=int(row['clf_v_ge_sig']),
                                           clf_v_ge_noi=int(row['clf_v_ge_noi']),
                                           clf_v_sy_sig=int(row['clf_v_sy_sig']),
                                           clf_v_sy_noi=int(row['clf_v_sy_noi']),

                                           # clf_t_g_size=int(row['clf_t_g_size']),
                                           # clf_t_s_size=int(row['clf_t_s_size']),
                                           # clf_v_g_size=int(row['clf_v_g_size']),
                                           # clf_v_s_size=int(row['clf_v_s_size']),
                                           model_base_file=str(model_base_file))
            results: Dict[str, float] = cp.execute()
            # update plan
            for metric in self.training_planner.metrics:
                if metric in results:
                    plan.at[index, metric] = results[metric]
            plan.at[index, 'done'] = 1.0
            plan.to_csv(self.cache_training_plan_file, index=False)

        print('____ RESULTS _____')
        print(plan)
        self.training_planner.plan = plan

    def __print_training_plan__(self):
        self.training_planner.print(self.result_training_plan)
        self.training_planner.print_confusion_matrices(self.result_confusion_matrices)

    def create_data_loader(self, norm_data: bool) -> DL2:
        raise NotImplementedError()

    def create_checkpoint_dir(self) -> Path:
        raise NotImplementedError()
