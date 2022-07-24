from __future__ import annotations

import math
import os

from common.globals import Global
from pathlib import Path

import pandas as pd
from tabulate import tabulate
from typing import List, Union, Optional, Dict, Any, Type

import numpy as np
import tensorflow as tf
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow import Tensor
from tensorflow.python.data import Dataset
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import Distribution as TD
from tensorflow_probability.python.bijectors import Bijector
from tensorflow_probability.python.bijectors import MaskedAutoregressiveFlow as TFMAF

from common import jsonloader, util
from common.util import Runtime
from common.NotProvided import NotProvided
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.kl.DivergenceMetric import DivergenceMetric
from maf.CustomMade import CustomMade
from broken.ClassOneHot import ClassOneHot
from maf.DS import DS, DSOpt, DSMethods, DataLoader
from distributions.base import cast_to_ndarray, TTensor, TDataOpt, TTensorOpt, MaybeBijKwargs, BaseMethods
from distributions.distribution import CutThroughData, HeatmapCreator
from distributions.density_plot_data import DensityPlotData
from distributions.LearnedDistribution import LearnedConfig, LearnedDistribution, EarlyStop, LearnedDistributionCreator
from maf.NoiseNormBijector import NoiseNormBijectorBuilder, NoiseNormBijector

"""(large) parts of this code are adapted from https://github.com/LukasRinder/normalizing-flows"""


class NanError(Exception):
    pass


class MafConfig(LearnedConfig):
    class Methods:
        @staticmethod
        def from_maf(maf: MaskedAutoregressiveFlow) -> MafConfig:
            c = MafConfig()
            c.js = maf.to_json(pretty_print=True)
            c.weights = [m.get_weights() for m in maf.transformed_distribution.submodules if hasattr(m, "get_weights")]
            return c

    def __init__(self):
        super().__init__()
        self.js: Optional[str] = None
        self.weights: List[np.ndarray] = []

    def create(self) -> MaskedAutoregressiveFlow:
        cp: MaskedAutoregressiveFlow = jsonloader.from_json(self.js)
        cp.build_transformation()

        # init everything by pushing through an example
        xs = np.ones(shape=(1, cp.input_dim))
        cond = None
        if cp.conditional:
            cond = np.array([cp.class_one_hot.classes[0]] * cp.conditional_dims, dtype=np.float32).reshape((1, cp.conditional_dims))
        cp.log_prob(xs=xs, cond=cond)

        # reset weights, because it behaved like an unlearned network without this in previous tests
        modules = [m for m in cp.transformed_distribution.submodules if hasattr(m, "get_weights")]
        for ws, m in zip(self.weights, modules):
            m.set_weights(ws)
        return cp


class MafHeatmapCreator(HeatmapCreator):

    def __init__(self, dist: LearnedDistribution):
        super().__init__(dist)

    def cut_along_x(self, pre_title: str, x_start: float = -4.0, x_end: float = 4.0, mesh_count: int = 1000) -> CutThroughData:
        if self.dist.input_dim != 2:
            raise RuntimeError("dimension != 2")
        dist: MaskedAutoregressiveFlow = self.dist
        x = tf.linspace(x_start, x_end, mesh_count)
        x = cast_to_ndarray(x)
        xs = x.reshape((mesh_count, 1))
        xs = np.column_stack([xs, np.zeros_like(xs)])
        u_x = dist.calculate_us(xs)
        u_x = np.array(u_x)
        det = dist.calculate_det_density(xs)
        det = np.array(det)
        ps_x = dist.prob(xs)
        c_x = CutThroughData(dimension_index=0, mesh_count=mesh_count, start=x_start, end=x_end, pre_title=pre_title, title_transform=f"transformation of t(x,0)", ps=ps_x, us=u_x,
                             det=det,
                             xs=x, x_label="x", f_name="(x,0)")
        return c_x

    def cut_along_y(self, pre_title: str, y_start: float = -4.0, y_end: float = 4.0, mesh_count: int = 1000) -> CutThroughData:
        if self.dist.input_dim != 2:
            raise RuntimeError("dimension != 2")
        dist: MaskedAutoregressiveFlow = self.dist
        y = tf.linspace(y_start, y_end, mesh_count)
        y = cast_to_ndarray(y)
        ys = y.reshape((mesh_count, 1))
        ys = np.column_stack([np.zeros_like(ys), ys])
        u_y = dist.calculate_us(ys)
        u_y = np.array(u_y)
        det = dist.calculate_det_density(ys)
        det = np.array(det)
        ps_y = dist.prob(ys)
        c_y = CutThroughData(dimension_index=1, mesh_count=mesh_count, start=y_start, end=y_end, pre_title=pre_title, title_transform=f"transformation of t(0,y)", ps=ps_y, us=u_y,
                             det=det,
                             xs=y, x_label="y", f_name="(0,y)")
        return c_y


class MaskedAutoregressiveFlow(LearnedTransformedDistribution):
    """
    Masked Autoregressive Flow implementation using Tensorflow probability.
    One can use self.transformed_distribution which transforms xs to us and vice versa
    while also taking care about the change of volume (aka Jacobian determinant).
    A few notes:
        - xs = bijector.forward(us) is the generating direction
        - us = bijector.inverse(xs) returns p(xs) WITHOUT undoing the volume change
        - bijector.[forward,inverse]_log_det_jacobian returns volume change. Multiply with xs, us respectively
        - the Maf permutes xs BEFORE feeding it to a MADE Layer
        - MADE layers is implemented in tfb.AutoregressiveNetwork
        - training/fit history is stored in this object but not returned on fit()
    """

    class Methods:
        @staticmethod
        def set_training(bijector: Bijector, training: bool):
            if isinstance(bijector, tfb.BatchNormalization):
                bijector.batchnorm.trainable = training
                bijector._training = training
                bijector.parameters["training"] = training
                pms: Dict[str, Any] = getattr(bijector, "_parameters")
                pms["training"] = training

        @staticmethod
        def deepcopy_maf(maf: MaskedAutoregressiveFlow) -> MaskedAutoregressiveFlow:
            config = MafConfig.Methods.from_maf(maf)
            cp = config.create()
            return cp

    def __init__(self, input_dim: int = NotProvided(), conditional_dims: int = 0, layers: int = NotProvided(), hidden_shape: List[int] = [200, 200], base_lr: float = 1e-3,
                 end_lr: float = 1e-4,
                 batch_norm: bool = False, norm_layer: bool = False, permutations: List[np.ndarray] = NotProvided(), use_tanh_made: bool = NotProvided(), build: bool = True,
                 activation: str = NotProvided(), input_noise_variance: float = 0.0, class_one_hot: ClassOneHot = NotProvided(), cond_type: str = NotProvided()):
        super().__init__(NotProvided.value_if_not_provided(input_dim, -1), conditional_dims=conditional_dims)
        self.layers: int = NotProvided.value_if_not_provided(layers, -1)
        self.class_one_hot: ClassOneHot = NotProvided.value_if_not_provided(class_one_hot, ClassOneHot(enabled=False))
        self.hidden_shape: List[int] = hidden_shape
        self.base_dist: Optional[tfd.Distribution] = None
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.input_noise_variance: float = input_noise_variance
        self.batch_norm: bool = batch_norm
        self.optimizer: Optional[OptimizerV2] = None
        self.permutations: List[np.ndarray] = NotProvided.value_if_not_provided(permutations, [])
        self.ignored.add("base_dist")
        self.ignored.add("optimizer")
        self.ignored.add("norm_layer_instance")
        self.ignored.add("built_transformation")
        self.norm_layer: bool = norm_layer
        self.norm_adapted: bool = False
        self.built_transformation: bool = False
        self.transformed_distribution: Optional[tfd.TransformedDistribution] = None
        self.use_tanh_made: bool = NotProvided.value_if_not_provided(use_tanh_made, False)
        self.activation: str = NotProvided.value_if_not_provided(activation, "relu")
        self.noise_norm_builder = NoiseNormBijectorBuilder(normalise=self.norm_layer, noise_stddev=input_noise_variance, batch_size=None)
        self.maf_layer_names: List[str] = []
        self.cond_type: Optional[str] = NotProvided.value_if_not_provided(cond_type, None)
        # if not self.norm_layer and NotProvided.is_provided(input_dim):
        #     self.build_transformation()

    def _create_base_distribution(self) -> Optional[TD]:
        self.build_transformation()
        return "placeholder, use transformed_distribution instead"

    def after_deserialize(self):
        # self.build_transformation()
        self.class_one_hot.init()

    def set_training(self, training: bool):
        for b in self.transformed_distribution.bijector.bijectors:
            MaskedAutoregressiveFlow.Methods.set_training(b, training)

    def _build_permutations(self):
        if len(self.permutations) == 0:
            # permutation = tf.cast(np.concatenate((np.arange(self.input_dim / 2, self.input_dim), np.arange(0, self.input_dim / 2))), tf.int32)
            permutation = tf.range(self.input_dim)
            permutation = tf.roll(permutation, math.floor(self.input_dim / 2), 0)
            new_permutation = permutation
            constant_permutation = tf.range(self.input_dim)
            for i in range(0, self.layers):
                self.permutations.append(cast_to_ndarray(new_permutation, dtype=np.int32))
                permutation = new_permutation
                while tf.reduce_all(new_permutation == constant_permutation) and self.input_dim > 1:
                    new_permutation = tf.random.shuffle(permutation)

    def build_transformation(self):
        if self.transformed_distribution is not None or self.built_transformation:
            return
        self.built_transformation = True
        self.base_dist: Optional[tfd.Distribution] = tfd.Normal(loc=0.0, scale=1.0)
        self.layers = Global.Testing.get('testing_nf_layers', self.layers)
        self.norm_layer = Global.Testing.get('testing_nf_norm_layer', self.norm_layer)
        print(
            f"building MAF: dim: {self.input_dim}, layers: {self.layers}, norm_layer: {self.norm_layer}, batch_norm: {self.batch_norm}, tahn_made: {self.use_tanh_made}, activation: {self.activation}")
        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=self.input_dim, dtype=tf.float32))
        self._build_permutations()
        bijectors: List[Bijector] = []
        counter = 0
        bijectors.append(self.noise_norm_builder.create())
        event_shape = (self.input_dim,)
        conditional_event_shape = (self.class_one_hot.output_dim(self.conditional_dims),) if self.conditional else None
        for i in range(0, self.layers):
            if self.input_dim > 1:
                bijectors.append(tfb.Permute(name=f"perm {i}", permutation=self.permutations[i]))
            maf_name = f"MAF_Layer_{i}"
            self.maf_layer_names.append(maf_name)

            if self.use_tanh_made:
                bijectors.append(tfb.MaskedAutoregressiveFlow(name=maf_name, shift_and_log_scale_fn=CustomMade(params=2, hidden_units=self.hidden_shape, activation=self.activation,
                                                                                                               conditional_event_shape=conditional_event_shape,
                                                                                                               conditional=self.conditional, event_shape=event_shape)))
            else:
                bijectors.append(
                    tfb.MaskedAutoregressiveFlow(name=maf_name,
                                                 shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2, hidden_units=self.hidden_shape, activation=self.activation,
                                                                                                  conditional_event_shape=conditional_event_shape, conditional=self.conditional,
                                                                                                  event_shape=event_shape)))
            counter += 1
            if counter % 2 == 0 and i + 1 < self.layers and self.batch_norm:
                bijectors.append(tfb.BatchNormalization(name=f"bn{i}"))
                counter = 0

        if self.batch_norm:
            bijectors.append(tfb.BatchNormalization(name="BEnd"))
        bijector = tfb.Chain(bijectors=bijectors, name="chain_of_maf")
        # self.transformed_distribution = tfd.TransformedDistribution(distribution=tfd.Sample(self.base_dist, sample_shape=[self.input_dim]),
        #                                                             bijector=bijector)
        self.transformed_distribution = tfd.TransformedDistribution(distribution=self.base_dist,
                                                                    bijector=bijector)

        self.set_training(False)
        if self.checkpoint_complete_prefix is not None:
            checkpoint = tf.train.Checkpoint(model=self.transformed_distribution)
            checkpoint.restore(self.checkpoint_complete_prefix)

    def adapt(self, dataset: Dataset):
        if self.norm_layer and not self.norm_adapted:
            print("adapting input normalization layer.")
            self.noise_norm_builder.adapt(dataset)
            print(
                f"norm layer: type={type(self.noise_norm_builder).__name__} shift {self.noise_norm_builder.norm_shift[:10]} scale {tf.exp(self.noise_norm_builder.norm_log_scale[:10])}")
            self.norm_adapted = True
            print("adapting done.")
            self.build_transformation()

    def _print_model(self):
        xx = np.ones((2, self.input_dim), dtype=np.float32)
        bijector_kwargs = None
        if self.conditional:
            cond = np.full((2, 1), self.class_one_hot.classes[0])
            bijector_kwargs: MaybeBijKwargs = self._create_bijector_kwargs(cond)
        self.transformed_distribution.log_prob(xx, bijector_kwargs=bijector_kwargs)
        rows = []

        for b in self.transformed_distribution.bijector.bijectors:
            # type[layer, dist], class, name, nodes, activation
            if isinstance(b, TFMAF):
                line = ['Layer', type(b).__name__, b.name, f"hs: {self.hidden_shape}", f"act: {self.activation}"]
            elif isinstance(b, NoiseNormBijector):
                line = ['Layer', type(b).__name__, b.name, f"noise: {b.noise_stddev > 0.0}", f"norm: {b.normalise}"]
            else:
                line = ['Layer', type(b).__name__, b.name, None, None]
            rows.append(line)
        rows.append(['Distr', type(self.base_dist).__name__, self.base_dist.name, None, None])
        maf_df: pd.DataFrame = pd.DataFrame(rows, columns=['type', 'class', 'name', 'params', 'params'])
        print(tabulate(maf_df, headers="keys", tablefmt="psql"))
        # sys.exit(8)

    def fit(self, dataset: [DS, DataLoader],
            epochs: int,
            batch_size: Optional[int] = None,
            val_xs: DSOpt = None,
            val_ys: TDataOpt = None,
            early_stop: Optional[EarlyStop] = None,
            plot_data_every: Optional[int] = None,
            lr: float = NotProvided(),
            val_contains_truth=False,
            shuffle: bool = False,
            divergence_metrics: Optional[List[DivergenceMetric]] = None,
            # ds_cond: DSOpt = None,
            # ds_val_cond: DSOpt = None
            ) -> Optional[List[DensityPlotData]]:
        if Global.Testing.has('testing_nf_norm_layer'):
            self.norm_layer = Global.Testing.get('testing_nf_norm_layer', self.norm_layer)
            if not self.norm_layer:
                self.noise_norm_builder.normalise = False
                self.build_transformation()
                self.set_training(True)
        ds_xs: DSOpt = None
        ds_val_xs: DSOpt = None
        ds_val_truth_exp: DSOpt = None
        ds_cond: DSOpt = None
        ds_val_cond: DSOpt = None
        if self.norm_layer and not self.norm_adapted:
            ds_xs, ds_val_xs, ds_cond, ds_val_cond, ds_val_truth_exp = self.fit_prepare(ds=dataset, batch_size=batch_size, val_ds=val_xs,
                                                                                        val_contains_truth=val_contains_truth, shuffle=shuffle, cond_cast_type=self.cond_type)
            self.adapt(ds_xs)
            self.set_training(True)
        else:
            ds_xs, ds_val_xs, ds_cond, ds_val_cond, ds_val_truth_exp = self.fit_prepare(ds=dataset, batch_size=batch_size, val_ds=val_xs,
                                                                                        val_contains_truth=val_contains_truth, shuffle=shuffle, cond_cast_type=self.cond_type)

        epochs = Global.Testing.get('testing_epochs', epochs)

        # print(f"MAF has {len(self.transformed_distribution.trainable_variables)} trainable variables")
        if epochs == 0:
            xx = np.ones((2, self.input_dim), dtype=np.float32)
            self.transformed_distribution.log_prob(xx)
            self.set_training(False)
            return []
        self.build_transformation()
        self._print_model()
        if epochs == -1:
            epochs = 99 * 99  # not infinite but big enough
            if early_stop is None:
                raise RuntimeError("epochs is set to -1 and EarlyStop is not provided. This would run forever.")
        # if ds_xs is None:
        #     ds_xs, ds_val_xs, ds_cond, ds_val_cond, ds_val_truth_exp = self.fit_prepare(ds=dataset, batch_size=batch_size, val_ds=val_xs,
        #                                                                                 val_contains_truth=val_contains_truth, shuffle=shuffle)
        self.set_training(True)
        learning_rate_fn = NotProvided.value_if_not_provided(lr, tf.keras.optimizers.schedules.PolynomialDecay(self.base_lr, epochs, self.end_lr, power=0.5))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        # import tensorflow.keras.optimizers as OOO
        # self.optimizer = OOO.Adadelta(lr=1.0)
        # self.optimizer = OOO.RMSprop()
        plot_data: List[DensityPlotData] = []
        runtime = Runtime("fit")
        # worked = False
        # nan_errors = 0

        ds_xs = DSMethods.batch(ds_xs, batch_size)
        ds_cond = DSMethods.batch(ds_cond, batch_size)
        ds_val_xs = DSMethods.batch(ds_val_xs, batch_size)
        ds_val_cond = DSMethods.batch(ds_val_cond, batch_size)
        ds_val_truth_exp = DSMethods.batch(ds_val_truth_exp, batch_size)

        util.p("stating fit...")
        epoch = 0
        # while not worked:
        if early_stop:
            early_stop = early_stop.new()
            early_stop.before_training_starts(self)
        try:
            for epoch in range(1, epochs + 1):
                runtime.start()
                batch_train_losses = []
                vs = zip(ds_xs, ds_cond) if self.conditional else zip(ds_xs, ds_xs)
                for xs, cond in vs:
                    if not self.conditional:
                        cond = None
                    bijector_kwargs = self._create_bijector_kwargs(cond, training=True)
                    train_loss = self.train_density_estimation(xs, bijector_kwargs)
                    batch_train_losses.append(train_loss)
                    if tf.math.is_nan(train_loss):
                        raise NanError(f"had a nan in epoch {epoch}")
                        # raise NanError()
                batch_val_losses = []
                # batch_val_kl_divs = []
                if ds_val_xs is not None:
                    # there are no Ys
                    if ds_val_truth_exp is None:
                        vs = zip(ds_val_xs, ds_val_cond) if self.conditional else zip(ds_val_xs, ds_val_xs)
                        for xs, cond in vs:
                            if not self.conditional:
                                cond = None
                            val_ps = self.log_prob(xs, cond=cond, batch_size=batch_size)
                            val_loss = -tf.reduce_mean(val_ps)
                            batch_val_losses.append(val_loss)
                    else:
                        # there are Ys. loop over them
                        vs = zip(ds_val_xs, ds_val_cond, ds_val_truth_exp) if self.conditional else zip(ds_val_xs, ds_val_xs, ds_val_truth_exp)
                        for d_xs, cond, d_ys_exp in vs:
                            if not self.conditional:
                                cond = None
                            val_ps = self.log_prob(d_xs, cond=cond)
                            # kl_divergence = tf.reduce_sum(d_ys_exp * (d_ys_exp - val_ps))
                            # batch_val_kl_divs.append(kl_divergence)
                            # self.history.add("kl", kl_divergence)
                            val_loss = -tf.reduce_mean(val_ps)
                            batch_val_losses.append(val_loss)

                train_loss = tf.reduce_mean(batch_train_losses)
                self.history.add("loss", train_loss)
                line = f"e {epoch} loss {train_loss} "
                if ds_val_xs is not None:
                    val_loss = tf.reduce_mean(batch_val_losses)
                    self.history.add("val_loss", val_loss)
                    line += f"val_loss {val_loss} "

                if divergence_metrics is not None:
                    for divergence_metric in divergence_metrics:
                        divergence_metric: DivergenceMetric = divergence_metric
                        divergence_metric.calculate(self.history, epoch)

                self.history.add('epoch', epoch)

                # if len(batch_val_kl_divs) > 0:
                #     kl = tf.reduce_mean(batch_val_kl_divs)
                #     self.history.add("kl", kl)
                #     line += f"kl {kl} "

                if early_stop is not None:
                    stop = early_stop.on_epoch_end(epoch, self.history)
                    if stop:
                        print(f"stopping after {epoch} epochs")
                        break

                runtime.stop()
                line += f"{runtime.to_string()} "
                print(line)
                runtime.reset()
                if plot_data_every is not None and epoch % plot_data_every == 0:
                    plot_data.append(self.heatmap_creator.heatmap_2d_data(title=f"after {epoch} epochs"))

        except NanError as e:
            print(f"error: {e}")
            if epoch < 2:
                raise e

        if early_stop:
            early_stop.after_training_ends(self.history)

            # except NanError:
            #     print("got NaN. restarting...", file=sys.stderr)
            #     nan_errors += 1
            #     self.create_base_distribution()
            #     self.set_training(False)
            #     if nan_errors == 10:
            #         raise NanError()
        self.set_training(False)
        if plot_data_every is not None:
            return plot_data

    def get_config(self) -> LearnedConfig:
        return MafConfig.Methods.from_maf(self)

    @tf.function
    def train_density_estimation(self, xs: TTensor, bijector_kwargs: MaybeBijKwargs) -> Dict[str, List[float]]:
        """
        Train function for density estimation normalizing flows.
        :param batch: Batch of the train data.
        :return: loss.
        """
        with tf.GradientTape() as tape:
            tape.watch(self.transformed_distribution.trainable_variables)
            ps = self.transformed_distribution.log_prob(xs, bijector_kwargs=bijector_kwargs)
            # ps = BaseMethods.un_nan(ps, replacement=tf.float32.min)
            # if tf.reduce_any(tf.math.is_nan(ps)):
            #     print('nan')
            loss = -tf.reduce_mean(ps)  # negative log likelihood
            # loss = -distribution.log_prob(batch)
            gradients = tape.gradient(loss, self.transformed_distribution.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.transformed_distribution.trainable_variables))
            reduced_loss = tf.reduce_mean(loss)
            return reduced_loss

    def _create_bijector_kwargs(self, cond: TTensorOpt = None, training: bool = False, sampling: bool = False) -> MaybeBijKwargs:
        if self.conditional and cond is None:
            raise ValueError("called without conditional variable")
        if self.conditional:
            cond = self.class_one_hot.encode(cond)
        bijector_kwargs: MaybeBijKwargs = {'conditional_input': cond, 'noise_norm': {'training': training}}
        for name in self.maf_layer_names:
            bijector_kwargs[name] = {'conditional_input': cond}
        return bijector_kwargs

    def _sample(self, size=1, cond: TTensorOpt = None, **kwargs) -> np.ndarray:
        sample_variance_multiplier = kwargs['sample_variance_multiplier'] if 'sample_variance_multiplier' in kwargs else 1.0
        self.set_training(False)
        bijector_kwargs = self._create_bijector_kwargs(cond)
        if sample_variance_multiplier != 1.0:
            us = self.base_dist.sample((size,)) * sample_variance_multiplier
            xs = self.transformed_distribution.bijector.forward(us, **bijector_kwargs)
            log_det = self.transformed_distribution.bijector.forward_log_det_jacobian(us, **bijector_kwargs)
            det = tf.exp(log_det)
            data = xs * tf.reshape(det, (len(det), 1))
            return cast_to_ndarray(data)
        else:
            samples = self.transformed_distribution.sample((size,), bijector_kwargs=bijector_kwargs)
            return cast_to_ndarray(samples)

    def _likelihoods(self, xs, cond: TTensorOpt = None) -> np.ndarray:
        xs, cond = self.extract_xs_cond(xs, cond)
        bijector_kwargs = self._create_bijector_kwargs(cond)
        probs: Tensor = self.transformed_distribution.prob(xs, bijector_kwargs=bijector_kwargs)
        probs = BaseMethods.un_nan(probs)
        return self.cast_2_likelihood(input_tensor=xs, result=probs)

    def _log_likelihoods(self, xs: TTensor, cond: TTensorOpt = None) -> np.ndarray:
        xs, cond = self.extract_xs_cond(xs, cond)
        bijector_kwargs = self._create_bijector_kwargs(cond)
        probs: Tensor = self.transformed_distribution.log_prob(xs, bijector_kwargs=bijector_kwargs)
        probs = BaseMethods.un_nan(probs, replacement=tf.float32.min)
        return self.cast_2_likelihood(result=probs, input_tensor=xs)

    def calculate_us(self, xs: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        xs, cond = self.extract_xs_cond(xs, cond)
        bijector_kwargs = self._create_bijector_kwargs(cond)
        us = self.transformed_distribution.bijector.inverse(xs, **bijector_kwargs)
        return us

    def calculate_xs(self, us: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        us, cond = self.extract_xs_cond(us, cond)
        bijector_kwargs = self._create_bijector_kwargs(cond)
        xs = self.transformed_distribution.bijector.forward(us, **bijector_kwargs)
        return xs

    def _calculate_det_density(self, xs: TTensor, cond: TTensorOpt = None) -> tf.Tensor:
        xs, cond = self.extract_xs_cond(xs, cond)
        bijector_kwargs = self._create_bijector_kwargs(cond)
        log_det = self.transformed_distribution.bijector.inverse_log_det_jacobian(xs, bijector_kwargs=bijector_kwargs)
        return tf.abs(tf.exp(log_det))

    @property
    def heatmap_creator(self) -> MafHeatmapCreator:
        return MafHeatmapCreator(self)

    @staticmethod
    def can_load_from(folder: Union[Path, str], prefix: str) -> bool:
        complete_prefix = str(Path(folder, prefix))
        json_file = f"{complete_prefix}.model.json"
        checkpoint_file = f"{complete_prefix}.data-00000-of-00001"
        return os.path.exists(json_file) and os.path.exists(checkpoint_file)

    @staticmethod
    def load(folder: Union[Path, str], prefix: str) -> MaskedAutoregressiveFlow:
        complete_prefix = str(Path(folder, prefix))
        json_file = f"{complete_prefix}.model.json"
        print(f"loading learned distribution from '{json_file}'")
        dis: MaskedAutoregressiveFlow = jsonloader.load_json(json_file)
        dis.checkpoint_complete_prefix = complete_prefix
        # checkpoint = tf.train.Checkpoint(model=dis.transformed_distribution)
        # checkpoint.restore(complete_prefix)
        # dis.set_training(False)
        return dis


class MAFCreator(LearnedDistributionCreator):
    def __init__(self, batch_norm: bool = NotProvided(),
                 # conditional_classes: List[str, int] = NotProvided(),
                 conditional_one_hot: bool = NotProvided(),
                 # epochs: int = NotProvided(),
                 hidden_shape: List[int] = NotProvided(),
                 input_noise_variance: float = NotProvided(),
                 layers: int = NotProvided(),
                 norm_layer: bool = NotProvided(),
                 use_tanh_made: bool = NotProvided()):
        super().__init__()
        self.batch_norm: bool = batch_norm
        # self.conditional_classes: List[str, int] = conditional_classes
        self.conditional_one_hot: bool = conditional_one_hot
        # self.epochs: int = epochs
        self.hidden_shape: List[int] = hidden_shape
        self.input_noise_variance: float = input_noise_variance
        self.layers: int = layers
        self.norm_layer: bool = norm_layer
        self.use_tanh_made: bool = use_tanh_made

    def create(self, input_dim: int, conditional_dims: int, conditional_classes: List[str, int] = None) -> LearnedDistribution:
        one_hot: ClassOneHot = ClassOneHot(enabled=self.conditional_one_hot, classes=conditional_classes, typ='int')
        noise_maf = MaskedAutoregressiveFlow(input_dim=input_dim, layers=self.layers, batch_norm=self.batch_norm,
                                             hidden_shape=self.hidden_shape,
                                             norm_layer=self.norm_layer, use_tanh_made=self.use_tanh_made, input_noise_variance=self.input_noise_variance,
                                             conditional_dims=conditional_dims, class_one_hot=one_hot)
        return noise_maf

    def load(self, folder: Union[Path, str], prefix: str) -> LearnedDistribution:
        maf = MaskedAutoregressiveFlow.load(folder=folder, prefix=prefix)
        return maf

    # def _save(self, dist: LearnedDistribution, folder: Union[Path, str], prefix: str):
    #     maf: MaskedAutoregressiveFlow = dist
    #     maf.save(folder=folder, prefix=prefix)
