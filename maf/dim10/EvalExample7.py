from keta.argparseer import ArgParser
from maf.dim10.EvalExample5 import EvalExample5
from maf.dim10.EvalLargeD import EvalLargeD


class EvalExample7(EvalExample5):
    def __init__(self):
        super().__init__('EvalExample7', layers=[3, 5, 7])


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample7().run()
