import argparse
from typing import Dict, Any

from common.globals import Global


class ArgParser:
    def __init__(self):
        self.ap: argparse.ArgumentParser = argparse.ArgumentParser()
        self.ap.add_argument('--gpu', help='select the gpu', default=0, type=int)

    def parse_args(self) -> Dict[str, Any]:
        args = vars(self.ap.parse_args())
        return args

    @staticmethod
    def parse():
        p = ArgParser()
        args = p.parse_args()
        Global.set_global('tf_gpu', args['gpu'])
