import argparse
from common.globals import Global


class ArgParser:
    @staticmethod
    def parse():
        p = argparse.ArgumentParser()
        p.add_argument('--gpu', help='select the gpu', default=0, type=int)
        args = vars(p.parse_args())
        Global.set_global('tf_gpu', args['gpu'])
