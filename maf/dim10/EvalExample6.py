from typing import List, Optional

from keta.argparseer import ArgParser
from maf.dim10.EvalLargeD import EvalLargeD


class EvalExample6(EvalLargeD):

    def __init__(self, name: str = 'EvalExample6', layers: Optional[List[int]] = None):
        self.loc_range = 4.0
        super().__init__(name, layers=layers)


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample6().run()
