from typing import Optional, List, Tuple

from keta.argparseer import ArgParser
from maf.visual_random.EvalLargeD import EvalLargeD


class EvalExample5(EvalLargeD):
    def __init__(self, name: str = 'EvalExample5', layers: Optional[List[int]] = None):
        super().__init__(name, layers=layers)


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample5().run()
