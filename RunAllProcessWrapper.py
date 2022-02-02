from __future__ import annotations

from common import jsonloader
from common.jsonloader import Ser
from importlib import import_module
from maf.stuff.MafExperiment import MafExperiment


class ProcessWrapper(Ser):
    class Methods:
        @staticmethod
        def static_execute(js: str):
            pw: ProcessWrapper = jsonloader.load_json(js)
            pw.run()

    def __init__(self, module: str, klass: str):
        super().__init__()
        self.module: str = module
        self.klass: str = klass

    def execute(self):
        js = jsonloader.load_json(self)
        ProcessWrapper.Methods.static_execute(js)

    def create_experiment(self) -> MafExperiment:
        mod = import_module(self.module)
        klass = getattr(mod, self.klass)
        return klass()

    def run(self):
        experiment: MafExperiment = self.create_experiment()
        experiment.run()
