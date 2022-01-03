from typing import List

from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.MultimodalDistribution import MultimodalDistribution


class GaussianMultimodal(MultimodalDistribution):
    """just for convenience"""

    def __init__(self, input_dim: int, distributions: List[GaussianMultivariate]):
        super().__init__(input_dim, [])
        self.distributions: List[GaussianMultivariate] = distributions