from __future__ import annotations

import math
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import tensorflow as tf
from tensorflow.python.data import Dataset

from common.util import Runtime
from common.NotProvided import NotProvided
from distributions.LearnedDistribution import LearnedDistribution, EarlyStop
from distributions.MultimodalDistribution import MultimodalDistribution
from distributions.Distribution import Distribution
from distributions.GaussianMultiModal import GaussianMultimodal
from distributions.GaussianMultivariate import GaussianMultivariate
from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow, NanError
from maf.SaveSettings import SaveSettings
from maf.pool import RestartingPoolReplacement


class Defaults:
    @staticmethod
    def create_default_model_definitions() -> List[Dict[str, Any]]:
        model_definitions = [{"model": "NF", "parameters": {"batch_size": 50, "layers": 1, "epochs": 200, "hidden_shape": [200, 200]}},
                             {"model": "NF", "parameters": {"batch_size": 50, "layers": 2, "epochs": 200, "hidden_shape": [200, 200]}},
                             {"model": "NF", "parameters": {"batch_size": 50, "layers": 3, "epochs": 200, "hidden_shape": [200, 200]}},
                             {"model": "NF", "parameters": {"batch_size": 50, "layers": 4, "epochs": 200, "hidden_shape": [200, 200]}}]
        return model_definitions

    @staticmethod
    def create_default_print_settings(prefix: Optional[str] = None) -> SaveSettings:
        return SaveSettings(ranges_start=[-13, -13], ranges_stop=[13, 13], step_size=0.05, prefix=prefix)

    @staticmethod
    def create_gauss_5_in_a_row_y() -> Distribution:
        source_distribution = GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-10, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[-5, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[5, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[10, 0], cov=[0.3, 0.3])])
        return source_distribution

    @staticmethod
    def create_gauss_5_in_a_row_x() -> Distribution:
        source_distribution = GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[0, -10], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, -5], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 0], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 5], cov=[0.3, 0.3]),
                                                                             GaussianMultivariate(input_dim=2, mus=[0, 10], cov=[0.3, 0.3])])
        return source_distribution

    @staticmethod
    def create_gauss_no() -> GaussianMultimodal:
        cov = [0.3, 0.3]
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[-10, -10], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[-7.5, 7.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[-5, -7.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[0, -2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[2.5, 5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[5, -5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[7.5, 0], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[10, 10], cov=cov)
        ])

    @staticmethod
    def create_gauss_no_4() -> GaussianMultimodal:
        cov = [0.3, 0.3]
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[0, -2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[2.5, 5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[5, -5], cov=cov),
        ])

    @staticmethod
    def create_gauss_no_3() -> GaussianMultimodal:
        cov = [0.3, 0.3]
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[0, -2.5], cov=cov),
            GaussianMultivariate(input_dim=2, mus=[2.5, 5], cov=cov)
        ])

    @staticmethod
    def create_gauss_4_diagonal() -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-7.5, 7.5], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[-2.5, 2.5], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[2.5, -2.5], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[7.5, -7.5], cov=[0.3, 0.3])])

    @staticmethod
    def create_1d_2_gauss():
        return MultimodalDistribution(input_dim=1, distributions=[GaussianMultivariate(input_dim=1, mus=[-2.5], cov=[1.0]),
                                                                  GaussianMultivariate(input_dim=1, mus=[2.5], cov=[1.0])])

    @staticmethod
    def create_gauss_1_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_2_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-2.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[2.5, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_3_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[5, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_4_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-7.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[-2.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[2.5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[7.5, offset], cov=[0.3, 0.3])])

    @staticmethod
    def create_gauss_5_y(offset: float) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[GaussianMultivariate(input_dim=2, mus=[-10, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[-5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[0, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[5, offset], cov=[0.3, 0.3]),
                                                              GaussianMultivariate(input_dim=2, mus=[10, offset], cov=[0.3, 0.3])])

    @staticmethod
    def swap_xy(multiple_nd_gaussians: GaussianMultimodal) -> GaussianMultimodal:
        gaussians = [GaussianMultivariate(input_dim=2, mus=[g.mus[1], g.mus[0]], cov=g.cov_matrix) for g in multiple_nd_gaussians.distributions]
        m = GaussianMultimodal(input_dim=2, distributions=gaussians)
        return m

    @staticmethod
    def create_biggus_dickus(offset: int = 0) -> GaussianMultimodal:
        return GaussianMultimodal(input_dim=2, distributions=[
            GaussianMultivariate(input_dim=2, mus=[0.0, offset], cov=[3, 3]),
            GaussianMultivariate(input_dim=2, mus=[-3, 3 + offset], cov=[2.0, 2.0]),
            GaussianMultivariate(input_dim=2, mus=[-3, -2 + offset], cov=[3.3, 2.3]),
            GaussianMultivariate(input_dim=2, mus=[4, 0 + offset], cov=[2.3, 3.3])
        ])


class VaryingParametersExperimentPlotter:
    def __init__(self, experiment: VaryingParametersExperiment):
        self.experiment: VaryingParametersExperiment = experiment
        sns.set_style("darkgrid")
        self.W = 12
        self.H = 7

    def print_cut_throughs(self):
        assert len(self.experiment.save_settings.ranges_stop) == len(self.experiment.save_settings.ranges_start) == 2

        def paint_transform(x_name: str, f_name: str, title: str, x: np.ndarray, u: np.ndarray):
            plt.clf()
            fig, axs = plt.subplots(2)
            fig.suptitle(title)
            fig.set_size_inches(self.W, self.H)
            fx = f"u_x{f_name}"
            fy = f"u_y{f_name}"
            f_names = [fx, fy]
            for d, ax in enumerate(axs):
                xx = np.column_stack([x, u[:, d]])
                f = f_names[d]
                df: pd.DataFrame = pd.DataFrame(xx, columns=[x_name, f])
                sns.lineplot(data=df, x=x_name, y=f, ax=ax)
            plt.savefig(f"{self.experiment.print_base_file_name}_{f_name}_transform.png")
            plt.clf()
            plt.close(fig)

        def paint(x_name: str, f_name: str, title: str, x: np.ndarray, u: np.ndarray, det: np.array()):
            # cut through, calculate Us, calculate likelihood
            fig, ax = plt.subplots()
            fig.suptitle(title)
            fig.set_size_inches(self.W, self.H)
            ps1 = multivariate_normal.pdf(u, cov=[1.0, 1.0], mean=[0.0, 0.0])
            xx = np.column_stack([x, ps1])
            df: pd.DataFrame = pd.DataFrame(xx, columns=[x_name, "Z"])
            sns.lineplot(data=df, x=x_name, y="Z")
            plt.savefig(f"{self.experiment.print_base_file_name}_{f_name}_CUT.png")
            plt.clf()
            plt.close(fig)

            # cut through, calculate Us, calculate likelihood and Jacobian, normalise
            fig, ax = plt.subplots(1)
            fig.suptitle(f"{title} |det(df/dx)|")
            fig.set_size_inches(self.W, self.H)
            ps1 = multivariate_normal.pdf(u, cov=[1.0, 1.0], mean=[0.0, 0.0])
            ps = ps1 * det
            xx = np.column_stack([x, ps])
            df: pd.DataFrame = pd.DataFrame(xx, columns=[x_name, "Z"])
            sns.lineplot(data=df, x=x_name, y="Z")
            plt.savefig(f"{self.experiment.print_base_file_name}_{f_name}_CUTDET.png")
            plt.clf()
            plt.close(fig)

        x = np.linspace(self.experiment.save_settings.ranges_start[0], self.experiment.save_settings.ranges_stop[0], 10000).reshape((10000, 1))
        vs = np.column_stack([x, np.zeros_like(x)])
        u = self.experiment.learned_distribution.calculate_us(vs)
        u = np.array(u)
        det = self.experiment.learned_distribution.calculate_det_density(vs)
        det = np.array(det)
        paint_transform(x_name="x", f_name="(x,0)", title=f"transformation of t(x,0)", x=x, u=u)
        paint(x_name="x", f_name="(x,0)", title="Z = p(f(x,0))", x=x, u=u, det=det)

        x = np.linspace(self.experiment.save_settings.ranges_start[1], self.experiment.save_settings.ranges_stop[1], 10000).reshape((10000, 1))
        vs = np.column_stack([np.zeros_like(x), x])
        u = self.experiment.learned_distribution.calculate_us(vs)
        u = np.array(u)
        det = self.experiment.learned_distribution.calculate_det_density(vs)
        det = np.array(det)
        paint_transform("y", f_name="(0,y)", title="transformation of t(0,y)", x=x, u=u)
        paint(x_name="y", f_name="(0,y)", title="Z = p(f(0,y))", x=x, u=u, det=det)

    def create_1d_plot(self, title: str):
        plt.clf()
        fig, ax = plt.subplots(1)
        fig.suptitle(title)
        fig.set_size_inches(18.5, 5)
        ax.title.set_text(f"{self.experiment.save_settings.get_prefix()}{title}")
        xs = np.linspace(-5.0, 5.0, 100).reshape((100, 1))
        y_truth = self.experiment.source_distribution.prob(xs).reshape((100, 1))
        y_model = self.experiment.learned_distribution.prob(xs).reshape((100, 1))

        xxs = np.concatenate([xs, xs])
        yys = np.concatenate([y_truth, y_model])
        types = np.concatenate([np.ones_like(xs), np.zeros_like(xs)])

        data = np.column_stack([xxs, yys, types])
        df: pd.DataFrame = pd.DataFrame(data, columns=["x", "y", "truth"])
        sns.lineplot(data=df, x="x", y="y", hue="truth")
        plt.savefig(f"{self.experiment.print_base_file_name}_1d.png")
        plt.clf()
        plt.close(fig)

    def create_data_3d_plot_fast(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("sampling for 3d data")
        r = Runtime("sampling").start()
        ps = self.experiment.save_settings
        step_size = ps.step_size
        from_coord = ps.ranges_start
        to_coord = ps.ranges_stop
        nr_z_0 = int((to_coord[0] - from_coord[0]) * (1 / step_size))
        nr_z_1 = int((to_coord[1] - from_coord[1]) * (1 / step_size))

        z = np.zeros((nr_z_0, nr_z_1))

        sh_0, sh_1 = z.shape
        x, y = np.linspace(from_coord[0], to_coord[0], sh_0), np.linspace(from_coord[1], to_coord[1], sh_1)
        X, Y = np.meshgrid(x, y)

        xx = X
        yy = Y
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack([r1, r2])
        p_truth = self.experiment.source_distribution.prob(grid)
        p_truth = p_truth.reshape(X.shape)
        p_truth = np.clip(p_truth, -10, math.inf)
        p = self.experiment.learned_distribution.prob(grid, batch_size=10000)
        p = p.reshape(X.shape)
        p = np.clip(p, -10, math.inf)
        r.stop().print()
        print("sampling for 3d data done")
        return x, y, p, p_truth

    def create_3d_plot_graphic(self, model_data, graphic_name):
        graphic_name = f"{self.experiment.save_settings.get_prefix()}{self.experiment.name}"
        x, y, p, p_truth = model_data
        fig = go.Figure(data=[go.Surface(z=p, x=x, y=y)])

        fig.update_layout(title=graphic_name, autosize=True,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.write_image(f"{self.experiment.print_base_file_name}_learned_density.png")
        # _write_image(fig, "ResultImages/3d_plot/", filename)

        fig = go.Figure(data=[
            go.Surface(z=p, x=x, y=y),
            go.Surface(z=p_truth, x=x, y=y, colorscale='Viridis', showscale=False, opacity=0.5)
        ])

        fig.update_layout(title=graphic_name + " Comparision", autosize=True,
                          margin=dict(l=65, r=50, b=65, t=90))
        ps = self.experiment.save_settings
        fig.write_image(f"{self.experiment.print_base_file_name}_comparison.png")
        fig.write_html(f"{self.experiment.print_base_file_name}_comparison.html")
        if self.experiment.save_settings.wants_3d_shown():
            fig.show()

    def scatter(self):
        ss: SaveSettings = self.experiment.save_settings
        fig, ax = plt.subplots()
        data = self.experiment.learned_distribution.sample(200)
        ax.set_xlim(ss.ranges_start[0], ss.ranges_stop[0])
        ax.set_ylim(ss.ranges_start[1], ss.ranges_stop[1])
        sns.scatterplot(x=data[:, 0], y=data[:, 1], size=0.01, ax=ax)
        plt.savefig(f"{self.experiment.print_base_file_name}_scatter.png")

    def plot(self):
        os.makedirs(self.experiment.save_settings.get_base_dir(), exist_ok=True)
        # diagrams
        if self.experiment.input_dim == 1:
            self.create_1d_plot(title=f"{self.experiment.model_type} {self.experiment.epochs}e")
        elif self.experiment.input_dim == 2:
            # 3d
            if self.experiment.save_settings.wants_3d_print():
                data = self.create_data_3d_plot_fast()
                self.create_3d_plot_graphic(model_data=data, graphic_name="debug 2")
            # scatter
            self.scatter()
            # f(y), f(x)
            self.print_cut_throughs()
        else:
            print(f"cannot print {self.experiment.input_dim}-dimensional plots", file=sys.stderr)


class ModelCreator:
    def __init__(self, d: Dict[str, Any], input_dim: int):
        self.d: Dict[str, Any] = d
        self.input_dim: int = input_dim
        self.defaults: Dict[str, Dict[str, Any]] = {
            "cnf":
                {"activation": "tanh",
                 "base_lr": 1e-3,
                 "batch_norm": False,
                 "batch_size": 50,
                 "conditional": True,
                 "conditional_dim": -1,
                 "end_lr": 10e-4,
                 "epochs": 50,
                 "hidden_shape": [1024, 1024],
                 "layers": 4,
                 "norm_layer": False,
                 "one_hot": False,
                 "use_tanh_made": False
                 },
            "nf":
                {"activation": "tanh",
                 "base_lr": 1e-3,
                 "batch_norm": False,
                 "batch_size": 50,
                 "conditional": False,
                 "conditional_dim": 0,
                 "end_lr": 10e-4,
                 "epochs": 50,
                 "hidden_shape": [1024, 1024],
                 "layers": 4,
                 "norm_layer": False,
                 "one_hot": False,
                 "use_tanh_made": False
                 }
        }

    def read_with_default_params(self, model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        d = self.defaults[model.lower()].copy()
        for k, v in params.items():
            if k not in d:
                raise ValueError(f"unknown parameter '{k}'")
            d[k] = v
        return d

    def create_model(self) -> LearnedDistribution:
        params: Dict[str, Any] = self.d.get("parameters", {})
        model_type: str = self.d.get("model")
        if model_type.lower() == "nf":
            params = self.read_with_default_params(model_type.lower(), params)
            activation: str = params.get("activation", "relu")
            base_lr: float = params.get("base_lr", 10e-3)
            batch_norm: bool = params.get("batch_norm", False)
            end_lr: float = params.get("end_lr", 10e-4)
            layers: int = params.get("layers", 4)
            hidden_shape: List[int] = params.get("hidden_shape", [200, 200])
            norm_layer: bool = params.get("norm_layer", False)
            use_tanh_made = params.get("use_tanh_made", False)
            maf = MaskedAutoregressiveFlow(input_dim=self.input_dim, layers=layers, hidden_shape=hidden_shape, norm_layer=norm_layer, batch_norm=batch_norm, base_lr=base_lr,
                                           end_lr=end_lr, activation=activation, use_tanh_made=use_tanh_made)
            return maf
        elif model_type.lower() == "cnf":
            raise NotImplementedError("Conditional MAF creation not implemented")
        else:
            raise NotImplementedError(f"creation of model type '{model_type}' not implemented")


class VaryingParametersExperiment:
    def __init__(self, model_definition: Dict[str, Any], source_distribution: Distribution, save_settings: SaveSettings, no_of_train_samples: int = 8000,
                 no_of_val_samples: int = 800, ):
        self.model_definition: Dict[str, Any] = model_definition
        self.no_of_train_samples: int = no_of_train_samples
        self.no_of_val_samples: int = no_of_val_samples
        self.source_distribution: Distribution = source_distribution
        self.input_dim: int = self.source_distribution.input_dim
        self.save_settings: SaveSettings = save_settings

        # created during runtime
        self.print_base_file_name: str = None
        self.train_samples: Optional[Dataset] = None
        self.val_samples: Optional[Dataset] = None
        self.val_prob_truth: Optional[Dataset] = None
        self.learned_distribution: Optional[LearnedDistribution] = None
        self.epochs: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.model_type: Optional[str] = None
        self.model_type: str = self.model_definition["model"].upper()
        # create model
        self.parameters: Dict[str, Any] = ModelCreator(self.model_definition, self.input_dim).read_with_default_params(self.model_definition["model"],
                                                                                                                       self.model_definition["parameters"])
        self.learned_distribution: MaskedAutoregressiveFlow = ModelCreator(self.model_definition, self.input_dim).create_model()
        self.name: str = f"{self.save_settings.get_prefix()}{self.learned_distribution.get_base_name_part(self.save_settings)}"
        self.print_base_file_name = f"{self.save_settings.get_base_dir()}{os.sep}{self.name}"

    def run(self, early_stop: Optional[EarlyStop]):
        self.base_dir = self.save_settings.get_base_dir()
        # generate samples
        train_samples = self.source_distribution.sample(self.no_of_train_samples)
        val_samples = self.source_distribution.sample(self.no_of_val_samples)
        val_prob_truth = self.source_distribution.log_prob(val_samples)
        self.train_samples = tf.data.Dataset.from_tensor_slices(train_samples)
        self.val_samples = tf.data.Dataset.from_tensor_slices(val_samples)
        self.val_prob_truth = tf.data.Dataset.from_tensor_slices(val_prob_truth)
        os.makedirs(self.save_settings.get_base_dir(), exist_ok=True)
        try:
            # train model
            if self.learned_distribution.can_load_from(self.save_settings.get_base_dir(), prefix=self.save_settings.get_prefix()):
                self.learned_distribution = self.learned_distribution.load(self.save_settings.get_base_dir(), prefix=self.save_settings.get_prefix())
            else:
                # if not self.learned_distribution.restore(self.print_base_file_name):
                parameters = self.parameters
                self.batch_size: int = parameters.get("batch_size", 50)
                self.epochs: int = parameters.get("epochs", 50)
                self.learned_distribution.fit(xs=self.train_samples, val_xs=self.val_samples, val_ys=self.val_prob_truth, epochs=self.epochs, batch_size=self.batch_size,
                                              early_stop=early_stop)
                self.learned_distribution.save(self.save_settings.get_base_dir(), prefix=self.save_settings.get_prefix())
                # save history
                hdf = pd.DataFrame(self.learned_distribution.history.to_dict())
                hdf.to_csv(f"{self.print_base_file_name}_history.csv")
            # diagrams & saving history
            plotter = VaryingParametersExperimentPlotter(self)
            plotter.plot()
        except NanError:
            print("got too many nan errors... skipping...")


class VaryingParametersExperimentSeries:
    class Methods:
        @staticmethod
        def start(source_distribution: Distribution, model_id: int, model_definition: Dict[str, Any], early_stop: Optional[EarlyStop],
                  print_settings: SaveSettings = SaveSettings()):
            name = f"{model_definition['model']}_{model_id}"
            e = VaryingParametersExperiment(name=name, model_definition=model_definition, source_distribution=source_distribution)
            e.run(early_stop=early_stop, save_settings=print_settings)

        @staticmethod
        def run_series(source_distribution: Distribution, model_definitions: List[Dict[str, Any]],
                       save_settings: SaveSettings = Defaults.create_default_print_settings(), early_stop: Optional[EarlyStop] = NotProvided(),
                       no_of_samples: int = 8000, no_of_val_samples: int = 800):
            early_stop = NotProvided.value_if_not_provided(early_stop, EarlyStop(monitor="val_loss", comparison_op=tf.less_equal, patience=10, restore_best_model=True))
            es = VaryingParametersExperimentSeries(source_distribution=source_distribution, model_definitions=model_definitions, early_stop=early_stop, print_settings=save_settings,
                                                   no_of_samples=no_of_samples, no_of_val_samples=no_of_val_samples)
            es.no_pool()
            es.run()

    def __init__(self, source_distribution: Distribution, model_definitions: List[Dict[str, Any]], early_stop: Optional[EarlyStop] = None,
                 print_settings: SaveSettings = SaveSettings(), no_of_samples: int = 8000, no_of_val_samples: int = 800):
        self.source_distribution: Distribution = source_distribution
        self.early_stop: Optional[EarlyStop] = early_stop
        self.model_definitions: List[Dict[str, Any]] = model_definitions
        self.save_settings: SaveSettings = print_settings
        self.no_of_samples: int = no_of_samples
        self.no_of_val_samples: int = no_of_val_samples
        self.use_pool: bool = True

    def no_pool(self) -> VaryingParametersExperimentSeries:
        """deactivate the use of a process pool. running in another process may break debugging.
        So if debugging does not work or acts weird try this."""
        self.use_pool = False
        return self

    def run(self):
        pool = RestartingPoolReplacement(processes=1)
        for model_id, model_definition in enumerate(self.model_definitions):
            if self.use_pool:
                pool.apply_async(VaryingParametersExperimentSeries.Methods.start, args=(self.source_distribution, model_id, model_definition, self.early_stop, self.save_settings))
            else:
                e = VaryingParametersExperiment(model_definition=model_definition, source_distribution=self.source_distribution, no_of_train_samples=self.no_of_samples,
                                                no_of_val_samples=self.no_of_val_samples, save_settings=self.save_settings)
                e.run(early_stop=self.early_stop.new())
        if self.use_pool:
            pool.join()


if __name__ == '__main__':
    early_stop = EarlyStop(monitor="val_loss", comparison_op=tf.less_equal, patience=10, restore_best_model=True)
    save_settings = SaveSettings(ranges_start=[-13, -13], ranges_stop=[13, 13], step_size=0.025, prefix="2dbig1024", print_3d=True, show_3d=True)
    model_definition = {"model": "NF", "parameters": {"activation": "relu",
                                                      "base_lr": 1e-3,
                                                      "batch_norm": False,
                                                      "batch_size": 128,
                                                      "conditional": False,
                                                      "conditional_dim": 0,
                                                      "end_lr": 10e-4,
                                                      "epochs": 5,
                                                      "hidden_shape": [1024, 1024],
                                                      "layers": 10,
                                                      "norm_layer": True,
                                                      "one_hot": False,
                                                      "use_tanh_made": True}}
    source_distribution = MultimodalDistribution(input_dim=2,
                                                 distributions=[GaussianMultivariate(input_dim=2, mus=[-10, -10], cov=[0.3, 0.3]),
                                                                GaussianMultivariate(input_dim=2, mus=[-10, -5], cov=[0.3, 0.3]),
                                                                GaussianMultivariate(input_dim=2, mus=[-10, 5], cov=[0.3, 0.3]),
                                                                GaussianMultivariate(input_dim=2, mus=[-10, 10], cov=[0.3, 0.3])])
    source_distribution = Defaults.create_gauss_no()
    # source_distribution = Defaults.create_gauss_no_4()
    # source_distribution = Defaults.create_gauss_no_3()
    # source_distribution = Defaults.create_1d_2_gauss()
    samples = source_distribution.sample(300)
    # plt.scatter(samples[:, 0], samples[:, 1])
    # plt.savefig("SCATR.png")
    e = VaryingParametersExperiment(model_definition=model_definition, source_distribution=source_distribution, no_of_train_samples=100000, no_of_val_samples=2000, save_settings=save_settings)
    e.run(early_stop=early_stop)
    print("exit")
