from __future__ import annotations

from pathlib import Path
from typing import Type, Optional
from unittest import TestCase

from keras import Input, Model
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.optimizer_v2.adam import Adam

from common import jsonloader
from common.NotProvided import NotProvided
from common.jsonloader import Ser
from distributions.kl.KL import KullbackLeiblerDivergence
from keta.lazymodel import LazyModel, LazyCustomObject


def create_model() -> Model:
    ins = Input(shape=(2,))
    b = Dense(512, activation='relu')(ins)
    # b = Dropout()
    b = Dense(512, activation='relu')(b)
    b = Dense(1, activation='sigmoid')(b)
    # b = Dense(100, activation='relu', name='DenseRELU0')(ins)
    # b = Dense(100, activation='relu')(b)
    # b = BatchNormalization()(b)
    # b = Dense(100, activation='relu')(b)
    # b = Dense(1, activation='linear', name='out')(b)
    model = Model(inputs=[ins], outputs=[b])
    return model
    lm = LazyModel.Methods.wrap(model)
    lm.compile(optimizer='adam', loss=BinaryCrossentropy(), lr=0.001, metrics=['accuracy'])
    return lm


class PlainDummy(Ser):
    def __init__(self):
        super().__init__()
        self.string: str = "test string"


class WithPath(Ser):
    def __init__(self, path: Path = NotProvided()):
        super().__init__()
        self.path: Path = path


class WithOptimzerObject(Ser):
    def __init__(self, opt: Adam = NotProvided()):
        super().__init__()
        self.opt: Adam = opt


class WithClass(Ser):
    def __init__(self):
        super().__init__()
        self.klass: Optional[Type[Ser]] = None


class JSTest(TestCase):
    def test_path(self):
        obj = WithPath(path=Path('/mnt'))
        js = jsonloader.to_json(obj, pretty_print=True)
        print(js)
        wp: WithPath = jsonloader.from_json(js)
        self.assertEqual(obj.path, wp.path)

    def test_model(self):
        oadam = LazyCustomObject.Methods.wrap('adam')
        adam: str = LazyCustomObject.Methods.unwrap(oadam)
        self.assertEqual('adam', adam)

    def test_class(self):
        obj = WithClass()
        obj.klass = KullbackLeiblerDivergence
        js = jsonloader.to_json(obj, pretty_print=True)
        des: WithClass = jsonloader.from_json(js)
        self.assertEqual(des.klass, obj.klass)
