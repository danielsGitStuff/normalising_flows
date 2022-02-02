from common.NotProvided import NotProvided
from maf.DL import DL2, DataSource
from pathlib import Path

from maf.mixlearn.dsinit.DSInitProcess import DSInitProcess


class DSBalanceInitProcess(DSInitProcess):
    def __init__(self, dl_cache_dir: Path = NotProvided(), experiment_cache_dir: Path = NotProvided(), test_split: float = 0.1):
        super().__init__(dl_cache_dir, experiment_cache_dir, test_split)

    def after_loading_main(self, main: DL2) -> DL2:
        target: Path = Path(main.dir, 'balanced')
        if DL2.can_load(target):
            return DL2.load(target)
        max_take = min(main.props.no_of_noise, main.props.no_of_signals)
        dl = DL2('asd',
                 dir=target,
                 signal_source=main.signal_source.ref(),
                 noise_source=main.noise_source.ref(),
                 amount_of_noise=max_take,
                 amount_of_signals=max_take)
        dl.create_data()
        return dl
