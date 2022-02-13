from common.NotProvided import NotProvided
from typing import List

from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution


class GaussianMultimodal(MultimodalDistribution):
    """just for convenience"""

    def __init__(self, input_dim: int = NotProvided(), distributions: List[GaussianMultivariate]=NotProvided()):
        super().__init__(input_dim, [])
        self.distributions: List[GaussianMultivariate] = distributions