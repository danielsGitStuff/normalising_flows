from pathlib import Path
from unittest import TestCase

from common import jsonloader
from common.NotProvided import NotProvided
from common.jsonloader import Ser


class Testi(Ser):
    def __init__(self, test: str = NotProvided()):
        super().__init__()
        self.test: str = NotProvided.value_if_not_provided(test, None)


class NotProvidedTest(TestCase):

    def test_with_None(self):
        obj = None
        self.assertFalse(NotProvided.is_not_provided(obj))
        self.assertTrue(NotProvided.is_provided(obj))

    def test_with_NP_instance(self):
        obj = NotProvided()
        self.assertTrue(NotProvided.is_not_provided(obj))
        self.assertFalse(NotProvided.is_provided(obj))

    def test_with_NP_class(self):
        obj = NotProvided
        self.assertTrue(NotProvided.is_not_provided(obj))
        self.assertFalse(NotProvided.is_provided(obj))

    def test_serialize_with_value(self):
        js = jsonloader.to_json(Testi(test="aaa"))
        testi: Testi = jsonloader.from_json(js)
        self.assertEqual(testi.test, 'aaa')

    def test_serialize_with_None(self):
        js = jsonloader.to_json(Testi(test=None))
        testi: Testi = jsonloader.from_json(js)
        self.assertIsNone(testi.test)

    def test_serialize_with_NP_class(self):
        js = jsonloader.to_json(Testi(test=NotProvided))
        testi: Testi = jsonloader.from_json(js)
        self.assertIsNone(testi.test)

    def test_serialize_with_NP_instance(self):
        js = jsonloader.to_json(Testi(test=NotProvided()))
        testi: Testi = jsonloader.from_json(js)
        self.assertIsNone(testi.test)
