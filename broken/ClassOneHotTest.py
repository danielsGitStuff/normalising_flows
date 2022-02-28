from __future__ import annotations

import tensorflow as tf
from typing import List

from keras.keras_parameterized import TestCase

from common import jsonloader
from distributions.base import TTensor
from broken.ClassOneHot import ClassOneHot


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