from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Collection

import numpy as np
import pandas as pd
import tensorflow as tf

from common import jsonloader
from common.NotProvided import NotProvided
from common.globals import Global
from common.jsonloader import Ser
from distributions.LearnedDistribution import EarlyStop, LearnedDistribution, LearnedDistributionCreator
from maf.ClassOneHot import ClassOneHot
from maf.DS import DS, DatasetProps
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
from maf.DL import DL2, DataSource


class MAFTrainingProcess(Ser):
    class Methods:
        @staticmethod
        def get_nf_file(cache_dir: Path, training_size: int, extension: str) -> Path:
            """@return the base file name for everything that just depends on a NF: sample file (.npy), nf file (.json)"""
            return Path(cache_dir, f"nf_{training_size}.{extension}")

        @staticmethod
        def get_classifier_name(cache_dir: Path, training_size: int, synth_ratio: float, extension: str) -> Path:
            """@return the base file name for everything that just depends on a classifier (witch itself depends on a NF): training history, model file"""
            return Path(cache_dir, f"cl_{training_size}_{synth_ratio}.{extension}")

    def __init__(self,
                 learned_distribution_creator: LearnedDistributionCreator = NotProvided(),
                 train_dir: Path = NotProvided(),
                 val_dir: Path = NotProvided(),
                 synth_dir: Path = NotProvided(),
                 synth_val_dir: Path = NotProvided(),
                 dl_init: DL2 = NotProvided(),
                 # training_size: int = NotProvided(),
                 epochs: int = NotProvided(),
                 batch_size: int = NotProvided(),
                 patience: int = NotProvided(),
                 # no_of_generated_samples: int = NotProvided(),
                 # no_of_generated_val_samples: int = NotProvided(),
                 conditional: bool = NotProvided(),
                 cache_dir: Path = NotProvided(),
                 checkpoint_dir: Path = NotProvided(),
                 checkpoint_dir_noise: Path = NotProvided(),
                 nf_base_file_name: str = NotProvided(),
                 # layers: int = NotProvided(),
                 # hidden_shape: List[int] = NotProvided(),
                 conditional_classes: Optional[Collection[int, str]] = None,
                 conditional_one_hot: bool = False,

                 size_nf_t_noi: int = NotProvided(),
                 size_nf_t_sig: int = NotProvided(),
                 size_nf_v_noi: int = NotProvided(),
                 size_nf_v_sig: int = NotProvided(),
                 gen_sig_samples: int = NotProvided(),
                 gen_noi_samples: int = NotProvided(),
                 gen_val_sig_samples: int = NotProvided(),
                 gen_val_noi_samples: int = NotProvided(),
                 take_t_sig: int = NotProvided(),
                 take_t_noi: int = NotProvided(),
                 take_v_sig: int = NotProvided(),
                 take_v_noi: int = NotProvided(),

                 # batch_norm: bool = False,
                 # norm_layer: bool = False,
                 # use_tanh_made: bool = False,
                 # input_noise_variance: float = 0.0,
                 val_size: int = 1500,
                 sample_variance_multiplier: float = 1.0):
        super().__init__()
        self.size_nf_t_noi: int = size_nf_t_noi
        self.size_nf_t_sig: int = size_nf_t_sig
        self.size_nf_v_noi: int = size_nf_v_noi
        self.size_nf_v_sig: int = size_nf_v_sig
        self.gen_sig_samples: int = gen_sig_samples
        self.gen_noi_samples: int = gen_noi_samples
        self.gen_val_sig_samples: int = gen_val_sig_samples
        self.gen_val_noi_samples: int = gen_val_noi_samples
        self.take_t_sig: int = take_t_sig
        self.take_t_noi: int = take_t_noi
        self.take_v_sig: int = take_v_sig
        self.take_v_noi: int = take_v_noi
        self.learned_distribution_creator: LearnedDistributionCreator = learned_distribution_creator
        self.train_dir: Path = train_dir
        self.val_dir: Path = val_dir
        self.synth_dir: Path = synth_dir
        self.synth_val_dir: Path = synth_val_dir
        self.sample_variance_multiplier: float = sample_variance_multiplier
        self.epochs: int = epochs
        self.patience: int = NotProvided.value_if_not_provided(patience, 10)
        self.batch_size: int = NotProvided.value_if_not_provided(batch_size, None)
        self.dl_init: DL2 = dl_init
        self.val_size: int = val_size

        # self.training_size: int = training_size
        # self.no_of_generated_samples: int = no_of_generated_samples
        # self.no_of_generated_val_samples: int = no_of_generated_val_samples
        # self.ds_synth_folder: Path = ds_synth_folder
        # self.ds_synth_val_folder: Path = ds_synth_val_folder
        self.conditional: bool = conditional
        self.conditional_classes: Optional[Collection[int, str]] = conditional_classes
        self.conditional_one_hot: bool = conditional_one_hot
        self.checkpoint_dir: Path = checkpoint_dir
        self.checkpoint_dir_noise: Path = checkpoint_dir_noise
        self.cache_dir: Path = cache_dir
        self.nf_base_file_name: str = NotProvided.none_if_not_provided(nf_base_file_name)
        # self.layers: int = NotProvided.none_if_not_provided(layers)
        # self.hidden_shape: List[int] = NotProvided.none_if_not_provided(hidden_shape)
        # self.batch_norm: bool = batch_norm
        # self.norm_layer: bool = norm_layer
        # self.use_tanh_made: bool = use_tanh_made
        # self.input_noise_variance: float = input_noise_variance
        self.dl_main: Optional[DL2] = None

    def create_dirs(self):
        def create(d: [Path, NotProvided]):
            if NotProvided.is_provided(d):
                if d is None:
                    return
                if isinstance(d, Path):
                    d.mkdir(exist_ok=True)
                else:
                    raise RuntimeError(f"unkndown thing: {type(d)}")

        create(self.checkpoint_dir)

    def after_deserialize(self):
        self.checkpoint_dir: Path = Path(self.checkpoint_dir)
        self.create_dirs()

    def run(self):
        self.create_dirs()
        one_hot = None
        conditional_dims = 0
        noise_maf: Optional[MaskedAutoregressiveFlow] = None
        lean_noise: bool = self.gen_noi_samples + self.gen_val_noi_samples > 0
        if self.conditional:
            # one_hot = ClassOneHot(enabled=self.conditional_one_hot, classes=self.conditional_classes, typ='int').init()
            conditional_dims = self.dl_init.conditional_dims
        if MaskedAutoregressiveFlow.can_load_from(self.checkpoint_dir, prefix=self.nf_base_file_name):
            maf: MaskedAutoregressiveFlow = MaskedAutoregressiveFlow.load(self.checkpoint_dir, prefix=self.nf_base_file_name)
            if lean_noise and not self.conditional:
                noise_maf: MaskedAutoregressiveFlow = MaskedAutoregressiveFlow.load(self.checkpoint_dir_noise, prefix=self.nf_base_file_name)
        else:
            dl_main_copy_dir = Path(self.cache_dir, 'dl_train')
            dl_main_test_dir = Path(self.cache_dir, 'dl_test')
            if not dl_main_copy_dir.exists() or not dl_main_test_dir.exists():
                self.dl_main = self.dl_init.clone(dl_main_copy_dir)
                dl_test = self.dl_main.split(test_dir=dl_main_test_dir, test_split=0.1)
            else:
                self.dl_main = DL2.load(dl_main_copy_dir)

            # If a saved train/test data set exists: use these.
            # Otherwise: load them from data loader, split them, save them and never load them from the data loader anymore.
            # Then: Split the training set into val/training and save them as well.
            if DL2.can_load(self.train_dir):
                dl_train = DL2.load(self.train_dir)
            else:
                # split_from_main_size = self.val_size + self.training_size
                # signal_noise_ratio = self.dl_main.amount_of_signals / (self.dl_main.amount_of_signals + self.dl_main.amount_of_noise)
                # take_signal = math.ceil(signal_noise_ratio * split_from_main_size)
                # take_noise = split_from_main_size - take_signal

                dl_train = DL2(dataset_name="asd",
                               dir=self.train_dir,
                               amount_of_noise=self.take_t_noi + self.take_v_noi,
                               amount_of_signals=self.take_t_sig + self.take_v_sig,
                               signal_source=self.dl_main.signal_source.ref(),
                               noise_source=self.dl_main.noise_source.ref())
                dl_train.create_data()

            if DL2.can_load(self.val_dir):
                dl_val: DL2 = jsonloader.load_json(Path(self.val_dir, 'dl2.json'))
            else:
                # dl_val = dl_train.split(test_dir=self.val_dir, test_amount=self.val_size)
                dl_val = dl_train.split2(test_dir=self.val_dir, take_test_sig=self.take_v_sig, take_test_noi=self.take_v_noi)

            if self.conditional:
                # signal_noise_ratio = self.dl_main.amount_of_signals / (self.dl_main.amount_of_signals + self.dl_main.amount_of_noise)
                # take_signal = math.ceil(signal_noise_ratio * self.training_size)
                # take_noise = self.training_size - take_signal
                # take_val_signal = math.ceil(signal_noise_ratio * self.val_size)
                # take_val_noise = self.val_size - take_val_signal
                ds_train = dl_train.get_conditional(amount_signal=self.size_nf_t_sig, amount_noise=self.size_nf_t_noi)
                ds_val = dl_val.get_conditional(amount_signal=self.size_nf_v_sig, amount_noise=self.size_nf_v_noi)
            else:
                ds_train = dl_train.get_signal(amount=self.size_nf_t_sig)
                ds_val = dl_val.get_signal(amount=self.size_nf_v_sig)
                if lean_noise and not self.conditional:
                    print('fitting noise')
                    ds_train_noise = dl_train.get_noise(amount=self.size_nf_t_noi)
                    ds_val_noise = dl_val.get_noise(amount=self.size_nf_v_noi)
                    # noise_maf = MaskedAutoregressiveFlow(input_dim=self.dl_main.props.dimensions, layers=self.layers, batch_norm=self.batch_norm,
                    #                                      hidden_shape=self.hidden_shape,
                    #                                      norm_layer=self.norm_layer, use_tanh_made=self.use_tanh_made, input_noise_variance=self.input_noise_variance,
                    #                                      conditional_dims=conditional_dims, class_one_hot=one_hot)
                    noise_maf = self.learned_distribution_creator.create(input_dim=self.dl_init.props.dimensions,
                                                                         conditional_dims=conditional_dims,
                                                                         conditional_classes=self.conditional_classes)
                    es = EarlyStop(monitor="val_loss", comparison_op=tf.less_equal, patience=self.patience, restore_best_model=True)
                    noise_maf.fit(ds_train_noise, epochs=self.epochs, batch_size=self.batch_size, val_xs=ds_val_noise, early_stop=es, shuffle=True)
                    noise_maf.save(folder=self.checkpoint_dir_noise, prefix=self.nf_base_file_name)
                    hdf = pd.DataFrame(noise_maf.history.to_dict())
                    hdf.to_csv(Path(self.cache_dir, f"{self.nf_base_file_name}_noise_history.csv"))
            # ds_train = ds_train.shuffle(buffer_size=len(ds_train), reshuffle_each_iteration=True)
            es = EarlyStop(monitor="val_loss", comparison_op=tf.less_equal, patience=self.patience, restore_best_model=True)
            # maf = MaskedAutoregressiveFlow(input_dim=self.dl_main.props.dimensions, layers=self.layers, batch_norm=self.batch_norm,
            #                                hidden_shape=self.hidden_shape,
            #                                norm_layer=self.norm_layer, use_tanh_made=self.use_tanh_made, input_noise_variance=self.input_noise_variance,
            #                                conditional_dims=conditional_dims, class_one_hot=one_hot)
            maf: LearnedDistribution = self.learned_distribution_creator.create(input_dim=self.dl_init.props.dimensions,
                                                                                conditional_classes=self.conditional_classes,
                                                                                conditional_dims=conditional_dims)
            maf.fit(dataset=ds_train, epochs=self.epochs, batch_size=self.batch_size, val_xs=ds_val, early_stop=es, shuffle=True)
            print(f"saving NF to '{self.checkpoint_dir}/{self.nf_base_file_name}'")
            maf.save(folder=self.checkpoint_dir, prefix=self.nf_base_file_name)
            hdf = pd.DataFrame(maf.history.to_dict())
            hdf.to_csv(Path(self.cache_dir, f"{self.nf_base_file_name}_history.csv"))

        if not DL2.can_load(self.synth_dir):
            print('sampling training data...')
            if lean_noise and not self.conditional:
                self.sample(maf=maf, synth_folder=self.synth_dir, gen_sig_samples=self.gen_sig_samples, gen_noi_samples=self.gen_noi_samples,
                            noise_source=DataSource(distribution=noise_maf))
            else:
                self.sample(maf=maf, synth_folder=self.synth_dir, gen_sig_samples=self.gen_sig_samples, gen_noi_samples=self.gen_noi_samples)
        if not DL2.can_load(self.synth_val_dir):
            print('sampling val data...')
            self.sample(maf=maf, synth_folder=self.synth_val_dir, gen_sig_samples=self.gen_val_sig_samples, gen_noi_samples=self.gen_val_noi_samples)

    def sample(self, maf: MaskedAutoregressiveFlow, synth_folder: Path, gen_sig_samples: int, gen_noi_samples: int, noise_source: DataSource = NotProvided()):
        noise_source: DataSource = NotProvided.value_if_not_provided(noise_source, self.dl_main.noise_source.ref())
        # p: DatasetProps = self.dl_main.props
        # signal_noise_ratio = p.no_of_signals / (p.no_of_signals + p.no_of_noise)
        # no_of_signals = math.ceil(no_of_samples * signal_noise_ratio)
        # no_of_noise = no_of_samples - no_of_signals
        if self.conditional:
            ones = np.ones(shape=(gen_sig_samples, 1), dtype=np.float32)
            zeros = np.zeros(shape=(gen_noi_samples, 1), dtype=np.float32)
            signals = maf.sample(gen_sig_samples, ones)
            noise = maf.sample(gen_noi_samples, zeros)
            signals = DS.from_tensor_slices(signals)
            noise = DS.from_tensor_slices(noise)
            dl = DL2(dataset_name="asdasd", dir=synth_folder,
                     signal_source=DataSource(ds=signals),
                     noise_source=DataSource(ds=noise),
                     amount_of_noise=gen_noi_samples,
                     amount_of_signals=gen_sig_samples)
            dl.create_data()
        else:
            dl = DL2(dataset_name='asd', dir=synth_folder,
                     signal_source=DataSource(distribution=maf),
                     noise_source=noise_source,
                     amount_of_signals=gen_sig_samples,
                     amount_of_noise=gen_noi_samples)
            dl.create_data()
        print('sampling done')

    @staticmethod
    def static_execute(mtp_js: str):
        # print('in new process')
        print('running in process')
        from distributions.base import enable_memory_growth
        enable_memory_growth()
        # print('des')
        mtp: MAFTrainingProcess = jsonloader.from_json(mtp_js)
        # print('dese')
        mtp.run()
        print('done with process')
        # print('ende!')

    def execute(self):
        js = jsonloader.to_json(self, pretty_print=True)
        # print('debug skip pool')
        # return MAFTrainingProcess.static_execute(js)
        Global.POOL().run_blocking(MAFTrainingProcess.static_execute, args=(js,))
