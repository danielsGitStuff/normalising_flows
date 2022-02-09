from typing import List, Optional

from keta.argparseer import ArgParser
from maf.dim10.EvalLargeD10 import EvalLargeD10


class Dim10cSmallGaps(EvalLargeD10):

    def __init__(self, name: str = 'Dim10cSmallGaps', layers: Optional[List[int]] = None, loc_range: float = 4.0):
        super().__init__(name, layers=layers, loc_range=loc_range)


if __name__ == '__main__':
    ArgParser.parse()
    Dim10cSmallGaps().run()
