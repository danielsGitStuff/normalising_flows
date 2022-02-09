from keta.argparseer import ArgParser
from maf.dim30.EvalLargeD30 import EvalLargeD30


class Dim30aLargeGaps(EvalLargeD30):
    def __init__(self):
        super().__init__('Dim30aLargeGaps')
        self.divergence_sample_size = 1024 * 300
        self.no_val_samples = 1024 * 8
        self.no_samples = 1024 * 800
        self.batch_size = 1024 * 4


if __name__ == '__main__':
    ArgParser.parse()
    Dim30aLargeGaps().run()
