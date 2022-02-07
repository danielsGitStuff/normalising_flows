from keta.argparseer import ArgParser
from maf.visual_random.EvalLargeD import EvalLargeD


class EvalExample6(EvalLargeD):

    def __init__(self):
        self.loc_range = 4.0
        super().__init__('EvalExample6')


if __name__ == '__main__':
    ArgParser.parse()
    EvalExample6().run()
