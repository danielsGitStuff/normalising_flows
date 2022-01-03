from __future__ import annotations

from typing import Dict, Any

from common.NotProvided import NotProvided
from maf.VaryingParametersExperiment import VaryingParametersExperimentSeries
from maf.VaryingParametersExperiment import Defaults as D
from maf.SaveSettings import SaveSettings


class ASeries:
    def __init__(self, model_definition: Dict[str, Any], no_of_samples: int = 8000, no_of_val_samples: int = 800):
        self.model_definition: Dict[str, Any] = model_definition
        self.no_of_samples: int = no_of_samples
        self.no_of_val_samples: int = no_of_val_samples

    def get_model_def(self) -> Dict[str, Any]:
        if NotProvided.is_not_provided(self.model_definition):
            model_definition = {"model": "NF", "parameters": {"batch_size": 50, "layers": 1, "epochs": 5000, "hidden_shape": [200, 200]}}
        else:
            model_definition = self.model_definition
        return model_definition

    def get_L(self) -> int:
        L = self.get_model_def()["parameters"]["layers"]
        return L

    def run(self):
        raise NotImplementedError()


class BiggusDickusSeries(ASeries):
    def __init__(self, model_definition: Dict[str, Any] = NotProvided(), no_of_samples: int = 8000, no_of_val_samples: int = 800):
        super().__init__(model_definition, no_of_samples, no_of_val_samples)

    def run(self):
        L = self.get_L()
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="BC", step_size=0.05, base_dir_ext=f"BC_L{L}", ranges_start=[-8, -8], ranges_stop=[8, 8]).print_3d(),
                                                             source_distribution=D.create_biggus_dickus(),
                                                             model_definitions=[self.get_model_def()])


class NonOccludedSeries(ASeries):
    def __init__(self, model_definition: Dict[str, Any] = NotProvided()):
        super().__init__(model_definition)

    def run(self):
        L = self.get_L()
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="NO", step_size=0.05, base_dir_ext=f"NO_L{L}").print_3d(),
                                                             source_distribution=D.create_gauss_no(),
                                                             model_definitions=[self.get_model_def()])


class OccludedSeries:
    def __init__(self, offset: float, model_definition: Dict[str, Any] = NotProvided()):
        self.offset: float = offset
        self.model_definition: Dict[str, Any] = model_definition

    def run(self):
        offset = self.offset

        if NotProvided.is_not_provided(self.model_definition):
            model_definition = {"model": "NF", "parameters": {"batch_size": 50, "layers": 1, "epochs": 5000, "hidden_shape": [200, 200]}}
        else:
            model_definition = self.model_definition

        L = model_definition["parameters"]["layers"]

        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="1G", step_size=0.05, base_dir_ext=f"Y_L{L}").print_3d(),
                                                             source_distribution=D.create_gauss_1_y(offset),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="2G", step_size=0.05, base_dir_ext=f"Y_L{L}").print_3d(),
                                                             source_distribution=D.create_gauss_2_y(offset),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="3G", step_size=0.05, base_dir_ext=f"Y_L{L}").print_3d(),
                                                             source_distribution=D.create_gauss_3_y(offset),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="4G", step_size=0.05, base_dir_ext=f"Y_L{L}").print_3d(),
                                                             source_distribution=D.create_gauss_4_y(offset),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="5G", step_size=0.05, base_dir_ext=f"Y_L{L}").print_3d(),
                                                             source_distribution=D.create_gauss_5_y(offset),
                                                             model_definitions=[model_definition])

        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="1Gx", step_size=0.05, base_dir_ext=f"X_L{L}").print_3d(),
                                                             source_distribution=D.swap_xy(D.create_gauss_1_y(offset)),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="2Gx", step_size=0.05, base_dir_ext=f"X_L{L}").print_3d(),
                                                             source_distribution=D.swap_xy(D.create_gauss_2_y(offset)),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="3Gx", step_size=0.05, base_dir_ext=f"X_L{L}").print_3d(),
                                                             source_distribution=D.swap_xy(D.create_gauss_3_y(offset)),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="4Gx", step_size=0.05, base_dir_ext=f"X_L{L}").print_3d(),
                                                             source_distribution=D.swap_xy(D.create_gauss_4_y(offset)),
                                                             model_definitions=[model_definition])
        VaryingParametersExperimentSeries.Methods.run_series(save_settings=SaveSettings(prefix="5Gx", step_size=0.05, base_dir_ext=f"X_L{L}").print_3d(),
                                                             source_distribution=D.swap_xy(D.create_gauss_5_y(offset)),
                                                             model_definitions=[model_definition])


if __name__ == '__main__':
    # s = OccludedSeries(offset=-6.5, model_definition={"model": "NF", "parameters": {"batch_size": 50, "layers": 1, "epochs": 5000, "hidden_shape": [200, 200]}})
    # s.run()
    # s = OccludedSeries(offset=-6.5, model_definition={"model": "NF", "parameters": {"batch_size": 50, "layers": 2, "epochs": 5000, "hidden_shape": [200, 200]}})
    # s.run()
    import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # tf.config.set_visible_devices([], 'GPU')
    for L in range(4, 5):
        # s = BiggusDickusSeries(model_definition={"model": "NF", "parameters": {"batch_size": 50, "layers": L, "epochs": 5000, "hidden_shape": [200, 200]}}, no_of_samples=16000,
        #                        no_of_val_samples=1600)
        s = NonOccludedSeries(
            model_definition={"model": "NF", "parameters": {"activation": "relu",
                                                            "base_lr": 1e-3,
                                                            "batch_norm": False,
                                                            "batch_size": 512,
                                                            "conditional": False,
                                                            "conditional_dim": 0,
                                                            "end_lr": 10e-4,
                                                            "epochs": 2,
                                                            "hidden_shape": [1024, 1024],
                                                            "layers": 10,
                                                            "norm_layer": True,
                                                            "one_hot": False,
                                                            "use_tanh_made": True}})
        # s = OccludedSeries(offset=0, model_definition={"model": "NF", "parameters": {"batch_size": 50, "layers": L, "epochs": 5000, "hidden_shape": [200, 200]}})
        s.run()
