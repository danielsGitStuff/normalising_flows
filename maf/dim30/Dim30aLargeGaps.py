from keta.argparseer import ArgParser
from maf.dim30.EvalLargeD30 import EvalLargeD30


class Dim30aLargeGaps(EvalLargeD30):
    def __init__(self):
        super().__init__('Dim30aLargeGaps')


if __name__ == '__main__':
    ArgParser.parse()
    Dim30aLargeGaps().run()
