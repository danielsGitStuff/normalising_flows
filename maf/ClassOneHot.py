from __future__ import annotations
from typing import Type, Collection, Union, List, Optional

from keras.keras_parameterized import TestCase
from keras.layers import CategoryEncoding, StringLookup, IntegerLookup
from keras.layers.preprocessing.index_lookup import IndexLookup
from tensorflow import Tensor

from common import jsonloader
from common.jsonloader import Ser
from common.NotProvided import NotProvided
from distributions.base import TTensor, cast_to_tensor
import tensorflow as tf


class ClassOneHot(Ser):
    def __init__(self, enabled: bool = NotProvided(), num_tokens: int = NotProvided(), typ: str = NotProvided(), classes: Collection[Union[str, int]] = NotProvided(),
                 mode: str = NotProvided()):
        super().__init__()
        self.enabled: bool = NotProvided.value_if_not_provided(enabled, False)
        self.classes: List[Union[str, int]] = sorted(set(NotProvided.value_if_not_provided(classes, [])))
        self.num_tokens: int = NotProvided.value_if_not_provided(num_tokens, len(self.classes))
        self.typ: str = NotProvided.value_if_not_provided(typ, None)
        self.mode: str = NotProvided.value_if_not_provided(mode, 'one_hot')
        self._indexer: Optional[IndexLookup] = None
        self._encoder: Optional[CategoryEncoding] = None
        self.ignored.add('_indexer')
        self.ignored.add('_encoder')

    def output_dim(self, input_dim: int) -> int:
        if not self.enabled:
            return input_dim
        return len(self.classes)

    def init(self) -> ClassOneHot:
        if not self.enabled:
            return self
        assert len(self.classes) > 0
        assert self.typ is not None
        if self.typ == "str":
            self._indexer = StringLookup(num_oov_indices=0)
        elif self.typ == "int":
            self._indexer = IntegerLookup(num_oov_indices=0)
        self._indexer.adapt(self.classes)
        self._encoder = CategoryEncoding(num_tokens=self.num_tokens, output_mode=self.mode)
        return self

    def encode(self, classes: TTensor) -> Tensor:
        if not self.enabled:
            t, _ = cast_to_tensor(classes)
            return t
        indexed = self._indexer(classes)
        result: Tensor = self._encoder(indexed)
        return result

    def after_deserialize(self):
        self.init()


class OneTest(TestCase):
    def setUp(self):
        self.ss: List[str] = ["a", "b", "c"]
        self.ii: List[int] = [1, 2, 3]
        self.sst: TTensor = tf.constant(self.ss, shape=(len(self.ss), 1))
        self.iit: TTensor = tf.constant(self.ii, shape=(len(self.ii), 1))
        self.hi = ClassOneHot(classes=self.ii, num_tokens=len(self.ii), typ='int', enabled=True).init()
        self.hs = ClassOneHot(classes=self.ss, num_tokens=len(self.ss), typ='str', enabled=True).init()
        self.hd = ClassOneHot(classes=self.ss, num_tokens=len(self.ss), typ='str', enabled=False).init()

    def test_ser(self):
        js = self.hi.to_json(pretty_print=True)
        print(js)
        cp: ClassOneHot = jsonloader.from_json(js)
        assert isinstance(cp, ClassOneHot)

    def test_ii(self):
        e = self.hi.encode(tf.constant([1, 2, 3], shape=(3, 1)))
        print(e)

    def test_disabled(self):
        self.assertAllEqual(self.sst, self.hd.encode(self.sst))
