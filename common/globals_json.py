from __future__ import annotations
from common import jsonloader
from pathlib import Path

from common.globals import Global
from common.jsonloader import Ser
from unittest import TestCase


class GlobalJson(Ser):
    @staticmethod
    def save() -> GlobalJson:
        g = GlobalJson()
        g.js_d_global = jsonloader.to_json(Global.d, pretty_print=True)
        g.js_d_testing = jsonloader.to_json(Global.Testing.d, pretty_print=True)
        return g

    def restore(self):
        Global.d = jsonloader.from_json(self.js_d_global)
        Global.Testing.d = jsonloader.from_json(self.js_d_testing)

    def __init__(self):
        super().__init__()
        self.js_d_global: str = 'not set'
        self.js_d_testing: str = 'not set'


class GlobalJsonTest(TestCase):
    def setUp(self):
        self.path: Path = Path('testing path')
        Global.set_global('test', True)
        Global.set_global('path', self.path)
        Global.Testing.set_global('testing_a', 'a')

    def test_ser(self):
        g = GlobalJson.save()
        print(g)
        Global.set_global('test', False)
        Global.set_global('path', 'i am a string now')
        Global.Testing.set_global('testing_a', 'b')
        g.restore()
        self.assertTrue(Global.d['test'])
        self.assertEqual(Global.d['path'], self.path)
        self.assertEqual(Global.Testing.d['testing_a'], 'a')
