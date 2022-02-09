from keta.argparseer import ArgParser
from maf.dim30.EvalLargeD30 import EvalLargeD30


class Dim30bSmallGaps(EvalLargeD30):
    def __init__(self):
        super().__init__('Dim30bSmallGaps', loc_range=5.0)


if __name__ == '__main__':
    ArgParser.parse()
    Dim30bSmallGaps().run()
