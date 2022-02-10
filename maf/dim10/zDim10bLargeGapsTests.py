from typing import Optional, List, Tuple

from keta.argparseer import ArgParser
from maf.dim10.EvalLargeD10 import EvalLargeD10


class zDim10bLargeGapsTest(EvalLargeD10):
    def __init__(self, name: str = 'zDim10bLargeGapsTest', layers: Optional[List[int]] = None, loc_range: float = 10.0):
        super().__init__(name, layers=[1], loc_range=loc_range)
        self.set_pool_size(2)


if __name__ == '__main__':
    ArgParser.parse()
    zDim10bLargeGapsTest().run()