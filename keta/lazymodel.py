from __future__ import annotations

import importlib
import os
import pathlib
import sys
from enum import Enum
from pydoc import locate
from typing import Tuple, Callable, Optional, List, Dict, Set, Any, Union
import numpy as np
import tensorflow as tf
from keras.layers import Normalization
from keras.models import Functional
from keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule
from keras.optimizer_v2.optimizer_v2 import OptimizerV2
from keras.utils.data_utils import Sequence
from numpy import ndarray
from tensorflow.keras import Model
from tensorflow.keras import layers as LLS
from tensorflow.keras.callbacks import History, Callback
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric, MeanAbsoluteError
from tensorflow.python.types.core import Tensor

from common import jsonloader, util
from common.NotProvided import NotProvided
from common.jsonloader import Ser
from maf.DS import DS, DSOpt


class LazyLayer(LLS.Layer, Ser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # super(LLS.Layer, self).__init__(**kwargs)
        # super(Ser, self).__init__()


class LazyField(Ser):
    class Methods:
        @staticmethod
        def class_path(object) -> str:
            t = type(object)
            d = vars(t)
            if "_keras_api_names" in d:
                return d["_keras_api_names"][0]
            klass = f"{t.__module__}.{t.__qualname__}"
            return klass

        @staticmethod
        def wrap(obj):
            if isinstance(obj, LazyField):
                return obj
            lf = LazyField()
            if isinstance(obj, Tensor):
                lf.klass = LazyField.Methods.class_path(obj)
                if lf.klass != 'tensorflow.python.framework.ops.EagerTensor':
                    lf.tf_name = obj.name
                lf.value = obj.numpy()
            elif isinstance(obj, Enum):
                lf.klass = LazyField.Methods.class_path(obj)
                lf.value = getattr(obj, "_value_")
                lf.enum = True
            elif isinstance(obj, Callable) and str(type(obj).__name__) == "function":
                lf.klass = "function"
                lf.d["m"] = obj.__module__  # module
                lf.d["n"] = obj.__qualname__.split(".")  # import name
            else:
                lf.klass = LazyField.Methods.class_path(obj)
                lf.value = obj
            return lf

        @staticmethod
        def locate(class_name: str):
            t = locate(class_name)
            if t is not None:
                return t
            if class_name.startswith("tensorflow.python.keras"):
                class_name = class_name[len("tensorflow.python."):]
                return LazyField.Methods.locate(class_name)
            raise RuntimeError(f"unknown class '{class_name}'")

        @staticmethod
        def unwrap(field):
            if not isinstance(field, LazyField):
                return field
            if field.enum:
                t = LazyField.Methods.locate(field.klass)
                e = t(field.value)
                return e
            if field.klass == "tensorflow.python.framework.ops.EagerTensor":
                t = tf.constant(field.value)
                return t
            if field.klass == "tensorflow.python.ops.resource_variable_ops.ResourceVariable":
                t = tf.Variable(field.value, name=field.tf_name)
                return t
            if field.klass == "builtins.NoneType":
                return None
            if field.klass == "function":
                module = field.d["m"]
                names = field.d["n"]
                module = importlib.import_module(module)
                for name in names:
                    module = getattr(module, name)
                return module  # is a function now

            # print(f"locate2 {field.klass}")
            t = locate(field.klass)
            if t is None:
                print("debug big error")
            # if t.__name__ == "LazyCustomObject":
            #     print(f"ttt {t}")
            result = t(field.value)
            return result

    def __init__(self):
        super().__init__()
        self.klass: str = None
        self.value: Any = None
        self.tf_name: str = None
        self.enum = False
        self.d: Dict[str, Any] = {}

    def to_string(self):
        return "{klass: {}, value: {}, tf_name: {}}".format(self.klass, self.value, self.tf_name)


class LazyCustomObject(Ser):
    class Methods:
        @staticmethod
        def wrap(obj: [Loss, Metric]) -> LazyCustomObject:
            if obj is None:
                return None
            if isinstance(obj, LazyCustomObject):
                return obj
            l = LazyCustomObject()
            l.set_object(obj)
            return l

        @staticmethod
        def wrap_to_dict(lazy_model, obj: [Loss, Metric, str, float, List, Dict]) -> LazyCustomObject:
            if isinstance(obj, LazyCustomObject) and obj.klass == "Dict":
                return obj
            lazy_model: LazyModel = lazy_model
            if not isinstance(obj, Dict):
                d = {output: LazyCustomObject.Methods.wrap(obj) for output in lazy_model.outputs}
                return LazyCustomObject.Methods.wrap(d)
            # assume existing dict is correct
            for output in lazy_model.outputs:
                if output not in obj:
                    raise RuntimeError(f"output '{output}' is not contained in the model")
            return LazyCustomObject.Methods.wrap(obj)

        @staticmethod
        def unwrap(obj: Any) -> Any:
            if isinstance(obj, LazyCustomObject):
                return obj.create()
            else:
                return obj

    def __init__(self):
        super().__init__()
        self.klass: str = None
        self.name: str = None
        self.value: Any = None
        self.object: [Loss, Metric] = None
        self.dict: Optional[Dict[str, LazyCustomObject]] = {}
        self.d: Dict[str, LazyField] = {}
        self.list: List[LazyCustomObject] = []
        self.ignored.add("object")

    def to_string(self):
        return "{klass: {}, name: {}, object: {}, dict: [{}], d: [{}]}".format(self.klass, self.name, self.object, self.dict, self.d)

    def create(self) -> [str, Loss, Dict]:
        if self.klass == "str":
            if self.value is not None:
                return self.value
            return self.object
        elif self.klass == "float":
            if self.value is not None:
                return self.value
            return self.object
        elif self.klass == "Dict":
            d: Dict[str, LazyCustomObject] = self.dict
            new_dict: Dict[str, LazyCustomObject] = {}
            for k, v in d.items():
                o: LazyCustomObject = v.create()
                new_dict[k] = o
            return new_dict
        elif self.klass == "list":
            ls = [LazyCustomObject.Methods.unwrap(o) for o in self.list]
            return ls
        elif self.klass == 'function':
            module = self.d["m"]
            names = self.d["n"]
            module = importlib.import_module(module)
            for name in names:
                module = getattr(module, name)
            return module  # is a function now
        else:
            # print(f"locate {self.klass}")
            t = locate(self.klass)
            obj: [Loss, Metric] = t()
            for k, lf in self.d.items():
                prop = LazyField.Methods.unwrap(lf)
                setattr(obj, k, prop)
            return obj

    def set_object(self, object: [str, Loss, Metric]):
        if isinstance(object, LazyCustomObject):
            self.set_object(object.object)
        elif isinstance(object, str):
            self.object = object
            self.value = object
            # self.name = object
            self.klass = "str"
        elif isinstance(object, float):
            self.value = object
            self.object = object
            self.klass = "float"
        elif isinstance(object, Dict):
            d = {}
            for k, o in object.items():
                lo = LazyCustomObject.Methods.wrap(o)
                d[k] = lo
            self.dict = d
            self.name = None
            self.klass = "Dict"
        elif isinstance(object, List):
            self.klass = "list"
            self.list = []
            for o in object:
                self.list.append(LazyCustomObject.Methods.wrap(o))
        elif isinstance(object, Callable) and str(type(object).__name__) == "function":
            # if t == "function":
            self.klass = "function"
            self.d["m"] = object.__module__  # module
            self.d["n"] = object.__qualname__.split(".")  # import name
            # else:
            #     raise RuntimeError(f"unknown function type '{t}'")
        else:
            self.object = object
            self.klass = LazyField.Methods.class_path(object)
            if hasattr(object,'name'):
                self.name = object.name
            # print(f"in goes {self.klass}")

            properties = {k: v for k, v in object.__dict__.items() if not k.startswith("_") and not k == "built" and not isinstance(v, Callable)}
            # if self.klass == "tensorflow.python.keras.metrics.MeanAbsoluteError":
            # print("debug sdfdsgtr")
            # for k, v in properties.items():
            #     print(f"{k} -> {v}")
            for k, p in properties.items():
                self.d[k] = LazyField.Methods.wrap(p)


if __name__ == '__main__':
    mae = MeanAbsoluteError()
    wrapped = LazyCustomObject.Methods.wrap(mae)
    unwrapped = LazyCustomObject.Methods.unwrap(wrapped)
    print(f"{wrapped}")


class LazyModel(Ser):
    class Methods:
        @staticmethod
        def base_to_h5_file(base_file: str):
            file = pathlib.Path(base_file)
            name = f"{file.name}.hdf5"
            h5 = file.with_name(name)
            return h5

        @staticmethod
        def extract_output_name(s: str) -> str:
            ss = s[:s.index("/")]
            return ss

        @staticmethod
        def base_to_js_file(base_file: str):
            file = pathlib.Path(base_file)
            name = f"{file.name}.json"
            js = file.with_name(name)
            return js

        @staticmethod
        def load_from_file(base_file: str):
            js_file = LazyModel.Methods.base_to_js_file(base_file)
            lm: LazyModel = jsonloader.load_json(js_file)
            lm.base_file = base_file
            return lm

        @staticmethod
        def model_exists(base_file: str) -> bool:
            js = f"{base_file}.json"
            return os.path.exists(js)

        @staticmethod
        def wrap(model: Model, base_file: str = None):
            l = LazyModel(base_file)
            l.model = model
            l.outputs = [LazyModel.Methods.extract_output_name(o.name) for o in model.outputs]
            l.inputs = [o.name for o in model.inputs]
            # append custom layers
            custom_layers = [layer for layer in model.submodules if isinstance(layer, LazyLayer)]
            l.custom_layers = LazyCustomObject.Methods.wrap(custom_layers)
            return l

        @staticmethod
        def unwrap_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            new_d: Dict[str, Any] = {}
            for k, v in d.items():
                if isinstance(v, List):
                    ls = [LazyCustomObject.Methods.unwrap(o) for o in v]
                    new_d[k] = ls
                elif isinstance(v, LazyCustomObject):
                    new_d[k] = LazyCustomObject.Methods.unwrap(v)
                else:
                    new_d[k] = v
            return new_d

    def __init__(self, base_file: str = None):
        super().__init__()
        self.base_file: str = base_file
        self.model: Functional = None
        self.outputs: List[str] = None
        self.inputs: List[str] = None
        self.ignored: Set[str] = self.ignored | {"model", "base_file"}

        self.optimizer: LazyCustomObject = LazyCustomObject()
        self.lr: LazyCustomObject = LazyCustomObject()
        self.losses: LazyCustomObject = LazyCustomObject()
        self.metrics: LazyCustomObject = LazyCustomObject()
        self.custom_layers: LazyCustomObject = LazyCustomObject()

    def clone(self, compile=True):
        import tensorflow as tf
        m = tf.keras.models.clone_model(self.model)
        wrapped: LazyModel = LazyModel.Methods.wrap(m, base_file=self.base_file)
        wrapped.model.build(self.model.layers[0].input.shape)
        if compile:
            wrapped.compile(optimizer=self.optimizer, lr=self.lr, loss=self.losses, metrics=self.metrics)
        wrapped.model.set_weights(self.model.get_weights())
        wrapped.optimizer = self.optimizer
        wrapped.lr = self.lr
        wrapped.metrics = self.metrics
        wrapped.losses = self.losses
        return wrapped

    def compile(self, optimizer: str = NotProvided(), lr: float = NotProvided(),
                loss: [Dict[str, str], Dict[str, Loss], Dict[str, LazyCustomObject]] = NotProvided(),
                metrics: Dict[str, List] = NotProvided(), testing_lr_schedule: LearningRateSchedule = NotProvided(), optimizer_rho: float = NotProvided()):
        self.eager_load()
        print(f"compiling model with optimizer '{optimizer}'")
        optimizer: str = LazyCustomObject.Methods.unwrap(optimizer)
        optimizer_rho: Optional[float] = LazyCustomObject.Methods.unwrap(optimizer_rho)
        optimizer_rho: float = NotProvided.value_if_not_provided(optimizer_rho, 0.95)
        lr: float = LazyCustomObject.Methods.unwrap(lr)
        # todo continue
        # lr : float = NotProvided.value_if_not_provided(lr, None)
        self.lr = LazyCustomObject.Methods.wrap(lr)
        if NotProvided.is_provided(optimizer):
            self.optimizer = LazyCustomObject.Methods.wrap(optimizer)
            # self.__wrap_to_output_dict(self.lr.dict, lr)
        if NotProvided.is_provided(metrics):
            self.set_metrics(metrics)
        if NotProvided.is_provided(loss):
            self.set_loss(loss)
        assert self.optimizer is not None
        assert self.lr is not None
        assert self.losses is not None
        lr = self.lr.create()
        losses = self.losses.create()
        metrics = self.metrics.create()

        from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Nadam, Adagrad
        opt_instance = None
        optimizer = optimizer.lower()
        if isinstance(lr, str) and lr.lower() == "default":
            lr = None
        testing_lr_schedule: Optional[LearningRateSchedule] = NotProvided.value_if_not_provided(testing_lr_schedule, None)
        if testing_lr_schedule is not None:
            lr = testing_lr_schedule
        if optimizer == "sgd":
            if lr is None:
                opt_instance = SGD()
            else:
                opt_instance = SGD(learning_rate=lr)
        if optimizer == "adam":
            if lr is None:
                opt_instance = Adam()
            else:
                opt_instance = Adam(learning_rate=lr)
        if optimizer == "adadelta":
            if lr is None:
                opt_instance = Adadelta(learning_rate=1.0)
            else:
                opt_instance = Adadelta(learning_rate=lr, rho=optimizer_rho)
        if optimizer == "nadam":
            if lr is None:
                base_lr = 0.001
                opt_instance = Nadam()
            else:
                opt_instance = Nadam(learning_rate=lr)
        if optimizer == "adagrad":
            if lr is None:
                opt_instance = Adagrad()
            else:
                opt_instance = Adagrad(learning_rate=lr)
        if optimizer == "rms":
            if lr is None:
                opt_instance = tf.keras.optimizers.RMSprop()
            else:
                opt_instance = tf.keras.optimizers.RMSprop(learning_rate=lr)
        if optimizer == "ftrl":
            if lr is None:
                opt_instance = tf.keras.optimizers.Ftrl()
            else:
                opt_instance = tf.keras.optimizers.Ftrl(learning_rate=lr)
        if isinstance(optimizer, OptimizerV2):
            opt_instance = optimizer
        if opt_instance is None:
            util.eprint("GOT WRONG OPTIMIZER! value was {}".format(optimizer))
        # if not metrics or len(metrics) == 0:
        #     metrics = [CountAcceptedMetric(), Correct(), Reward(), "accuracy"]

        for k, ms in metrics.items():
            print(f"     COMPILE OUTPUT {k}")
            print(f"     COMPILE METRICS {[m if isinstance(m, str) else m.name for m in ms]}")
        if "out_goals" not in self.get_outputs():
            if "out_goals" in losses:
                del losses["out_goals"]
            if "out_goals" in metrics:
                del metrics["out_goals"]
        self.model.compile(loss=losses,  # BetHelper.bet_loss, # run no 1,2
                           optimizer=opt_instance, metrics=metrics)  # ,

    def eager(self, compile: bool = True):
        self.eager_load()
        if compile:
            self.compile()

    def eager_load(self):
        if not self.model:
            model_file = LazyModel.Methods.base_to_h5_file(self.base_file)
            import tensorflow.keras as keras
            custom_objects = {}
            metrics = LazyCustomObject.Methods.unwrap(self.metrics)
            losses = LazyCustomObject.Methods.unwrap(self.losses)
            for k, m in metrics.items():
                custom_objects[k] = m
            for k, l in losses.items():
                custom_objects[l.name] = l
            # custom layers
            custom_layers: List[LazyLayer] = self.custom_layers.create()
            for layer in custom_layers:
                t = type(layer)
                custom_objects[t.__qualname__] = t
            if len(custom_objects) > 0:
                self.model = keras.models.load_model(model_file, custom_objects=custom_objects, compile=False)
            else:
                self.model = keras.models.load_model(model_file)
            # fix a bug present in tf 2.6.0 where normalization layer has the learned weights set but ignores them
            for layer in self.model.submodules:
                if type(layer) == Normalization:
                    layer: Normalization = layer
                    layer.set_weights(layer.get_weights())

    def fit_generator(self, gen: Sequence, val_gen: Sequence = None, batch_size=None, epochs=None, verbose=0, callbacks: Optional[Dict[str, List[Callable]]] = None) -> History:
        cbs: Dict[str, Callback] = {out: callbacks[out] for out in self.outputs}
        history = self.model.fit(gen, validation_data=val_gen, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=cbs,
                                 shuffle=True)
        return history

    def prepare_ds(self, ds: DSOpt, conditional_dims: int) -> DSOpt:
        if ds is None:
            return None
        cond = ds.map(lambda t: t[:conditional_dims], num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda t: t[conditional_dims:], num_parallel_calls=tf.data.AUTOTUNE)
        # cond = ds.map(lambda t: t[:conditional_dims])
        # ds = ds.map(lambda t: t[conditional_dims:])
        ds: DS = tf.data.Dataset.zip((ds, cond))
        return ds

    def predict_data_set(self, ds: DS, conditional_dims: int, batch_size: Optional[int] = NotProvided()):
        ds = self.prepare_ds(ds, conditional_dims=conditional_dims)
        batch_size = NotProvided.value_if_not_provided(batch_size, len(ds))
        ds = ds.batch(batch_size=batch_size)
        ps = self.model.predict(x=ds)
        return ps

    def evaluate_data_set(self, ds: DS, conditional_dims: int, batch_size: Optional[int] = NotProvided()) -> Tuple[List[float], List[str]]:
        ds = self.prepare_ds(ds, conditional_dims=conditional_dims)
        batch_size = NotProvided.value_if_not_provided(batch_size, len(ds))
        ds = ds.batch(batch_size=batch_size)
        ps = self.model.evaluate(x=ds)
        return ps, self.model.metrics_names

    def fit_data_set(self, ds_train: DS, conditional_dims: int, ds_val: DSOpt = None, batch_size=None, epochs: int = None, verbose=0,
                     callbacks: Optional[Dict[str, List[Callable]]] = None, shuffle: bool = False) -> History:
        """This is adapted to MAF stuff!!! Conditional dims come first here!!!"""
        if conditional_dims < 1:
            raise ValueError("cannot fit without having a 'y'")

        ds_train = self.prepare_ds(ds_train, conditional_dims=conditional_dims)
        ds_val = self.prepare_ds(ds_val, conditional_dims=conditional_dims)

        if shuffle:
            ds_train = ds_train.shuffle(buffer_size=len(ds_train), reshuffle_each_iteration=True)
        if batch_size is None:
            batch_size = len(ds_train)
        ds_train = ds_train.batch(batch_size=batch_size)
        ds_val = ds_val.batch(batch_size=batch_size)
        h = self.model.fit(x=ds_train, validation_data=ds_val, epochs=epochs, verbose=verbose, callbacks=callbacks)
        return h

    def fit(self, xs: ndarray, ys: ndarray, batch_size=None, epochs=None, verbose=0,
            callbacks: Optional[List[Callable]] = None,
            validation_data: Optional[Tuple[ndarray, ndarray]] = None, validation_split=0.0, shuffle: bool = True,
            sample_weights: ndarray = None) -> History:
        self.eager_load()
        if validation_data and len(validation_data[0]) == 0:
            validation_data = None
        if isinstance(xs, tf.data.Dataset):
            history = self.model.fit(xs, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=validation_data,
                                     validation_split=validation_split, shuffle=shuffle, sample_weight=sample_weights)
        else:
            # callbacks = []
            if "out_goals" not in self.get_outputs():
                del ys["out_goals"]
                del callbacks["out_goals"]
                if validation_data is not None:
                    del validation_data[1]["out_goals"]
            history = self.model.fit(xs, ys, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=validation_data,
                                     validation_split=validation_split, shuffle=shuffle, sample_weight=sample_weights)
        return history

    def get_metric_from_history(self, history: [History, Dict[str, Any]], out_name: str, metric: str):
        if isinstance(history, History):
            history = history.history
        key = f"{out_name}_{metric}"
        v = history[key]
        return v

    def get_outputs(self) -> List[str]:
        return self.outputs

    def is_single_output(self):
        if self.model is None or len(self.model.outputs) == 0:
            raise RuntimeError("no model or model has no outputs")
        return len(self.model.outputs) == 1

    def save(self):
        h5_file = LazyModel.Methods.base_to_h5_file(self.base_file)
        js_file = LazyModel.Methods.base_to_js_file(self.base_file)
        # for _ in range(3):
        #     util.eprint("DEBUG: LayzModel.save() DOES NOT SAVE THE ACTUAL MODEL")
        from tensorflow import keras
        keras.models.save_model(self.model, h5_file)
        jsonloader.to_json(self, js_file, pretty_print=True)
        print("debug oaooasf")

    def set_loss(self, losses: [[Loss], Dict[str, Loss]]):
        self.losses = LazyCustomObject.Methods.wrap_to_dict(self, losses)
        # self.losses = LazyCustomObject.Methods.wrap(losses)
        # self.__wrap_to_output_dict(self.losses.dict, losses)

    def set_metrics(self, metrics: [List[Metric], Dict[str, List[Metric]]]):
        self.metrics = LazyCustomObject.Methods.wrap_to_dict(self, metrics)
        # self.__wrap_to_output_dict(self.metrics.dict, metrics)
        # self.metrics = [LazyCustomObject.Methods.wrap(m) for m in metrics]

    def predict(self, xs: ndarray):
        if len(xs) == 0:
            return np.array([])
        self.eager_load()
        probs = self.model.predict(xs)
        return probs

    def predict3(self, xs: ndarray):
        probs = self.predict(xs).tolist()
        preds = [[0, 0, 0] for ps in probs]
        for pre, pro in zip(preds, probs):
            pre[np.argmax(pro)] = 1
        maxProbs = [float(np.amax(ps)) for ps in probs]
        return probs, preds, maxProbs

    def print_status(self):
        raise NotImplementedError()

    def __wrap_to_output_dict(self, target_dict: Dict[str, LazyCustomObject], obj: Any):
        def extract_first_output_name() -> str:
            return LazyModel.Methods.extract_output_name(self.model.outputs[0].name)

        if self.is_single_output():
            if isinstance(obj, Dict):
                if len(obj) != 1:
                    raise ValueError(f"dict has {len(obj)} entries but must have 1")
                target_dict[extract_first_output_name()] = LazyCustomObject.Methods.wrap([obj.values()][0])
            else:
                target_dict[extract_first_output_name()] = obj
        else:
            if isinstance(obj, Dict):
                for output, o in obj.items():
                    target_dict[output] = LazyCustomObject.Methods.wrap(o)
            else:
                for output in self.model.outputs:
                    target_dict[LazyModel.Methods.extract_output_name(output.name)] = LazyCustomObject.Meth


class LazyReference:
    def __init__(self, base_file: Union[str, pathlib.Path]):
        self.base_file: Union[str, pathlib.Path] = base_file


class LazyMethods:
    @staticmethod
    def ref(lazy_model: LazyModel) -> LazyReference:
        lazy_model.save()
        r = LazyReference(lazy_model.base_file)
        return r

    @staticmethod
    def unref(r: LazyReference) -> LazyModel:
        lm = LazyModel.Methods.load_from_file(r.base_file)
        lm.unwrap()
        return lm
