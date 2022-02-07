from keta.argparseer import ArgParser
from maf.visual_random.EvalLargeD import EvalLargeD


class EvalExample5(EvalLargeD):
    def __init__(self):
        super().__init__('EvalExample5')


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample5().run()
