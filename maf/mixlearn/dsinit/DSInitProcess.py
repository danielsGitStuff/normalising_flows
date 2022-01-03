from __future__ import annotations

from pathlib import Path

from common import jsonloader
from common.NotProvided import NotProvided
from common.globals import Global
from common.jsonloader import Ser
from maf.DL import DL2


class DSInitProcess(Ser):
    @staticmethod
    def execute(p: DSInitProcess):
        js = jsonloader.to_json(p, pretty_print=True)
        # print('debug skip pool')
        # return DSInitProcess.execute_static(js)
        Global.POOL().run_blocking(DSInitProcess.execute_static, args=(js,))

    @staticmethod
    def execute_static(js: str):
        p: DSInitProcess = jsonloader.from_json(js)
        p.run()

    def __init__(self, dl_cache_dir: Path = NotProvided(), experiment_cache_dir: Path = NotProvided(), test_split: float = 0.1):
        super().__init__()
        self.dl_cache_dir: Path = dl_cache_dir
        self.experiment_cache_dir: Path = experiment_cache_dir
        self.test_split: float = test_split

    @property
    def train_dir(self) -> Path:
        return Path(self.experiment_cache_dir, 'dl_train')

    @property
    def test_dir(self) -> Path:
        return Path(self.experiment_cache_dir, 'dl_test')

    def run(self):
        if self.train_dir.exists() and self.test_dir.exists():
            self.after_initialisation()
            return
        dl_main = DL2.load(self.dl_cache_dir)
        dl_train = dl_main.clone(self.train_dir)
        dl_test = dl_train.split(self.test_dir, test_split=self.test_split)
        self.after_initialisation()

    def after_initialisation(self):
        """you may modify the training or test set here.
        CHECK IF YOU MUST RUN HERE by looking for a file or something. """
        pass
