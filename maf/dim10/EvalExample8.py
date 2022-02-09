from keta.argparseer import ArgParser
from maf.dim10.Dim10cSmallGaps import Dim10cSmallGaps


class EvalExample8(Dim10cSmallGaps):
    def __init__(self):
        super().__init__('EvalExample8', layers=[3, 5, 7, 10, 20, 30])


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample8().run()
