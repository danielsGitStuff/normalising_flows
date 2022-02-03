from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from common import jsonloader
from common.NotProvided import NotProvided
from common.globals import Global
from common.jsonloader import Ser
from importlib import import_module

from distributions.base import set_gpu, enable_memory_growth
from maf.stuff.MafExperiment import MafExperiment
from maf.stuff.StaticMethods import StaticMethods


class GPUProcessWrapperPool(Ser):
    class Methods:
        @staticmethod
        def static_launch(js: str):
            wp: GPUProcessWrapperPool = jsonloader.from_json(js)
            wp.run_on_current_gpu()

    def __init__(self):
        super().__init__()
        self.d: Dict[int, List[GPUProcessWrapper]] = {}
        self.current_gpu: int = 0

    def after_deserialize(self):
        self.d: Dict[int, List[GPUProcessWrapper]] = {int(k): ls for k, ls in self.d.items()}

    def add_to_pool(self, pw: GPUProcessWrapper, gpu: int):
        if gpu not in self.d:
            self.d[gpu] = []
        pw.gpu = gpu
        self.d[gpu].append(pw)

    def launch(self):
        if len(self.d.keys()) > 1:
            for gpu in self.d.keys():
                self.current_gpu = gpu
                js = jsonloader.to_json(self, pretty_print=True)
                Global.POOL().apply_async(GPUProcessWrapperPool.Methods.static_launch, args=(js,))
                # GPUProcessWrapperPool.Methods.static_launch(js)
            Global.POOL().join()
        else:
            gpu = list(self.d.keys())[0]
            for pw in self.d[gpu]:
                pw.execute()

    def run_on_current_gpu(self):
        Global.set_global('tf_gpu', self.current_gpu)
        # enable_memory_growth()
        for pw in self.d[self.current_gpu]:
            pw.execute()


class GPUProcessWrapper(Ser):
    class Methods:
        @staticmethod
        def static_execute(js: str):
            pw: GPUProcessWrapper = jsonloader.from_json(js)
            pw.run()

    def __init__(self, module: str = NotProvided(), klass: str = NotProvided(), results_dir: Union[Path, str] = NotProvided()):
        super().__init__()
        self.module: str = module
        self.klass: str = klass
        self.results_dir: Union[Path, str] = results_dir

    def execute(self):
        js = jsonloader.to_json(self, pretty_print=True)
        # print('debug skip process')
        # return ProcessWrapper.Methods.static_execute(js)
        Global.POOL().run_blocking(GPUProcessWrapper.Methods.static_execute, args=(js,))

    def create_experiment(self) -> MafExperiment:
        mod = import_module(self.module)
        klass = getattr(mod, self.klass)
        return klass()

    def run(self):
        cache = StaticMethods.cache_dir()
        Global.set_global('results_dir', Path(self.results_dir))
        check_file: Path = Path(cache, f"done_{self.module}.{self.klass}")
        if check_file.exists():
            print(f"experiment '{self.module}.{self.klass}' already done. skipping...")
            return
        print(f"launching experiment '{self.module}.{self.klass}'")
        experiment: MafExperiment = self.create_experiment()
        experiment.run()
        check_file.touch()
