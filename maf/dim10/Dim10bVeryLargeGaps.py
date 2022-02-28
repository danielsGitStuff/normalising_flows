from typing import Optional, List, Tuple

from common.argparser import ArgParser
from maf.dim10.EvalLargeD10 import EvalLargeD10


class Dim10bVeryLargeGaps(EvalLargeD10):
    def __init__(self, name: str = 'Dim10bVeryLargeGaps', layers: Optional[List[int]] = None, loc_range: float = 15.0):
        super().__init__(name, layers=layers, loc_range=loc_range)


if __name__ == '__main__':
    ArgParser.parse()
    Dim10bVeryLargeGaps().run()
