from typing import List

import numpy as np

from common.jsonloader import Ser
from distributions.Distribution import Distribution
from distributions.LearnedDistribution import FitHistory
from distributions.kl.Divergence import Divergence
from distributions.kl.JS import JensenShannonDivergence
from distributions.kl.KL import KullbackLeiblerDivergence
from maf.DS import DS


class DivergenceMetric(Ser):
    def __init__(self, maf: Distribution, ds_samples: DS, log_ps_samples: DS, run_every_epoch: int = 2, batch_size: int = 100000):
        super().__init__()
        self.maf: Distribution = maf
        self.run_every_epoch: int = run_every_epoch
        self.batch_size: int = batch_size
        self.divergences: List[Divergence] = [KullbackLeiblerDivergence(p=self.maf, q=self.maf, half_width=0.0, step_size=666.0, batch_size=self.batch_size),
                                              JensenShannonDivergence(p=self.maf, q=self.maf, half_width=0.0, step_size=666.0, batch_size=self.batch_size)]
        self.ds_samples: DS = ds_samples
        self.log_ps_samples: DS = log_ps_samples

    def calculate(self, fit_history: FitHistory, epoch: int):
        if epoch % self.run_every_epoch != 0 and epoch > 1:
            for d in self.divergences:
                fit_history.add(d.name, None)
            return
        for d in self.divergences:
            result = d.calculate_from_samples_vs_p(ds_q_samples=self.ds_samples, log_q_samples=self.log_ps_samples)
            fit_history.add(d.name, result)
