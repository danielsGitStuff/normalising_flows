import sys
from time import sleep

import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('module')
    p.add_argument('klass')
    p.add_argument('gpu', type=int)
    args = vars(p.parse_args())
    print(args)
    # sleep(10)
    # sys.exit(0)
    from importlib import import_module
    from common.globals import Global
    from maf.stuff.MafExperiment import MafExperiment

    Global.set_global('tf_gpu', args['gpu'])
    mod = import_module(args['module'])
    klass = getattr(mod, args['klass'])
    ex: MafExperiment = klass()
    ex.run()
    # print(type(ex))
