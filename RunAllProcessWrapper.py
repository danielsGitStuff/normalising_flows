from __future__ import annotations

import setproctitle as setproctitle

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
            pool = Global.NEW_POOL(3)
            for gpu in self.d.keys():
                self.current_gpu = gpu
                js = jsonloader.to_json(self, pretty_print=True)
                pool.apply_async(GPUProcessWrapperPool.Methods.static_launch, args=(js,))
                # GPUProcessWrapperPool.Methods.static_launch(js)
            pool.join()
        else:
            gpu = list(self.d.keys())[0]
            # for pw in self.d[gpu]:
            #     pw.execute()
            p = Global.NEW_POOL()
            for pw in self.d[self.current_gpu]:
                js = jsonloader.to_json(pw)
                print('addddding')
                p.apply_async(GPUProcessWrapper.Methods.static_execute, args=(js,))
                # GPUProcessWrapper.Methods.static_execute(js)
            p.join()

    def run_on_current_gpu(self):
        setproctitle.setproctitle(f"Pool {self.current_gpu}")
        Global.set_global('tf_gpu', self.current_gpu)
        # p = Global.NEW_POOL()
        for pw in self.d[self.current_gpu]:
            js = jsonloader.to_json(pw)
            print('addddding')
            Global.POOL().run_blocking(GPUProcessWrapper.Methods.static_execute, args=(js,))
            # p.apply_async(GPUProcessWrapper.Methods.static_execute,args=(js,))
            # GPUProcessWrapper.Methods.static_execute(js)
        # p.join()
        # for pw in self.d[self.current_gpu]:
        #     pw.execute()


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
        setproctitle.setproctitle(f"W {Global.get_default('tf_gpu', -1)} {self.klass} {Global.get_default('tf_gpu', -1)}")
        cache = StaticMethods.cache_dir()
        if NotProvided.is_provided(self.results_dir) and self.results_dir is not None:
            Global.set_global('results_dir', Path(self.results_dir))
        check_file: Path = Path(cache, f"done_{self.module}.{self.klass}")
        if check_file.exists():
            print(f"experiment '{self.module}.{self.klass}' already done. skipping...")
            return
        print(f"launching experiment '{self.module}.{self.klass}'")
        experiment: MafExperiment = self.create_experiment()
        experiment.run()
        check_file.touch()
