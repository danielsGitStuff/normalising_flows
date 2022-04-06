from typing import List

import numpy as np
import scipy.stats

from distributions.Distribution import Distribution


class ShiftAndScale2(Distribution):
    def __init__(self, input_dim: int, shift: List[float], scale: List[float], data: Distribution):
        super().__init__(input_dim)
        self.shift: np.ndarray = np.array(shift, dtype=np.float32)
        self.scale: np.ndarray = np.array(scale, dtype=np.float32)
        self.data: Distribution = data
        # self.data = scipy.stats.multivariate_normal(mean=[0] * self.input_dim, cov=[1] * self.input_dim)

    def xs_to_us(self, xs: np.ndarray) -> np.ndarray:
        return self.scale * (xs + self.shift)

    def us_to_xs(self, us: np.ndarray) -> np.ndarray:
        return (us - self.shift) / self.scale

    def prob(self, xs: np.ndarray, batch_size: int = None) -> np.ndarray:
        us = self.us_to_xs(xs)
        ps = self.data.prob(us) * self.xs_to_us_jacobian_det()
        ps = self.cast_2_likelihood(xs, ps)
        return ps

    def xs_to_us_jacobian_det(self) -> float:
        return np.prod(1 / self.scale)


class ShiftAndScale2Wrong(ShiftAndScale2):
    def __init__(self, input_dim: int, shift: float, scale: float, data: Distribution):
        super().__init__(input_dim, shift, scale, data)

    def xs_to_us_jacobian_det(self) -> float:
        return 1.0
