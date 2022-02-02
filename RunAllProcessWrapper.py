from __future__ import annotations

from pathlib import Path

from common import jsonloader
from common.NotProvided import NotProvided
from common.globals import Global
from common.jsonloader import Ser
from importlib import import_module
from maf.stuff.MafExperiment import MafExperiment
from maf.stuff.StaticMethods import StaticMethods


class ProcessWrapper(Ser):
    class Methods:
        @staticmethod
        def static_execute(js: str):
            pw: ProcessWrapper = jsonloader.from_json(js)
            pw.run()

    def __init__(self, module: str = NotProvided(), klass: str = NotProvided()):
        super().__init__()
        self.module: str = module
        self.klass: str = klass

    def execute(self):
        js = jsonloader.to_json(self, pretty_print=True)
        # print('debug skip process')
        # return ProcessWrapper.Methods.static_execute(js)
        Global.POOL().run_blocking(ProcessWrapper.Methods.static_execute, args=(js,))

    def create_experiment(self) -> MafExperiment:
        mod = import_module(self.module)
        klass = getattr(mod, self.klass)
        return klass()

    def run(self):
        cache = StaticMethods.cache_dir()
        check_file: Path = Path(cache, f"done_{self.module}.{self.klass}")
        if check_file.exists():
            print(f"experiment '{self.module}.{self.klass}' already done. skipping...")
            return
        print(f"launching experiment '{self.module}.{self.klass}'")
        experiment: MafExperiment = self.create_experiment()
        experiment.run()
        check_file.touch()
