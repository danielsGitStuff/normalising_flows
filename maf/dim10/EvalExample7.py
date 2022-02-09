from keta.argparseer import ArgParser
from maf.dim10.Dim10bLargeGaps import Dim10bLargeGaps
from maf.dim10.EvalLargeD10 import EvalLargeD10


class EvalExample7(Dim10bLargeGaps):
    def __init__(self):
        super().__init__('EvalExample7', layers=[3, 5, 7, 10, 20, 30])


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample7().run()
