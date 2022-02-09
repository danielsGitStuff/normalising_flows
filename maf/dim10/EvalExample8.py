from keta.argparseer import ArgParser
from maf.dim10.EvalExample6 import EvalExample6


class EvalExample8(EvalExample6):
    def __init__(self):
        super().__init__('EvalExample8', layers=[3, 5, 7])


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample8().run()
