from __future__ import annotations

import sys

from maf.DL import DL3
from maf.dry_examples.EvalExample1 import EvalExample1
from maf.dry_examples.EvalExample2 import EvalExample2
from maf.dry_examples.EvalExample4 import EvalExample4
from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunnerBalanced import MixLearnExperimentMiniBooneClfVarRunnerBalanced
from maf.mixlearn.dl3.MinibooneDL3 import MinibooneDL3
from maf.visual_examples.NF1D_1Bumps import NF1D_1Bumps
from maf.visual_examples.NF1D_2Bumps import NF1D_2Bumps
from maf.visual_examples.NF2D_10Bumps import NF2D_10Bumps
from maf.visual_examples.NF2D_1Bumps import NF2D_1Bumps
from maf.visual_examples.NF2D_1Rect import NF2D_1Rect
from maf.visual_examples.NF2D_2Bumps import NF2D_2Bumps
from maf.visual_examples.NF2D_3Rect import NF2D_3Rect
from maf.visual_examples.NF2D_4Rect import NF2D_4Rect
from maf.visual_examples.NF2D_Diag4 import NF2D_Diag4
from maf.visual_examples.NF2D_Row3 import NF2D_Row3
from maf.visual_examples.SS1DExample import SS1DMafExperiment
from maf.visual_examples.SS2DExample import SS2DMafExperiment
from maf.visual_examples.ShowCase1D1 import ShowCase1D1
from pathlib import Path

from RunAllProcessWrapper import GPUProcessWrapper, GPUProcessWrapperPool
from keta.argparseer import ArgParser
from maf.dry_examples.EvalExample3 import EvalExample3
from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner import MixLearnExperimentMiniBooneClfVarRunner

from common.globals import Global
from typing import List, Type, Dict, Any

from maf.visual_examples.NF2D_Row4 import NF2D_Row4

if __name__ == '__main__':
    # ArgParser.parse()
    ap = ArgParser()
    ap.ap.add_argument('--big_machine', help='use more GPUs', action='store_true')
    args: Dict[str, Any] = ap.parse_args()
    description: str = f"""
    This script runs all MAF experiments inlcuding learning artificial distributions like multivariate/multimodal Gaussians, Miniboone, MNIST and experiments to show divergence.

    """
    print(description)
    gpu_pool = GPUProcessWrapperPool()


    def run(examples: List[Type], results_dir: Path, gpu: int = 0):
        for t in examples:
            pw = GPUProcessWrapper(module=t.__module__, klass=t.__name__, results_dir=results_dir)
            gpu_pool.add_to_pool(pw, gpu)


    examples_artificial: List[Type] = [NF1D_1Bumps,
                                       NF1D_2Bumps,
                                       NF2D_1Bumps,
                                       NF2D_2Bumps,
                                       NF2D_10Bumps,
                                       NF2D_1Rect,
                                       NF2D_3Rect,
                                       NF2D_4Rect,
                                       SS1DMafExperiment,
                                       SS2DMafExperiment,
                                       ShowCase1D1,
                                       NF2D_Diag4,
                                       NF2D_Row3,
                                       NF2D_Row4
                                       ]

    examples_dry: List[Type] = [EvalExample1,
                                EvalExample2,
                                EvalExample3,
                                EvalExample4
                                ]

    # this speeds up training!
    # Global.Testing.set('testing_epochs', 1)
    # Global.Testing.set('testing_nf_layers', 1)
    # Global.Testing.set('testing_nf_norm_layer', False)

    examples_mix_learn: List[Type] = [MixLearnExperimentMiniBooneClfVarRunner,
                                      MixLearnExperimentMiniBooneClfVarRunnerBalanced]

    # make sure the dataset is already in place before starting processes relying on it. Might cause race conditions otherwise
    test_pool = GPUProcessWrapperPool()
    pw = GPUProcessWrapper(module=MinibooneDL3.__module__, klass=MinibooneDL3.__name__, results_dir='nanana useless')
    test_pool.add_to_pool(pw, 2)
    test_pool.add_to_pool(GPUProcessWrapper(module=NF2D_3Rect.__module__, klass=NF2D_3Rect.__name__, results_dir='results_artificial'), 2)
    test_pool.launch()
    sys.exit(9)
    # pw.execute()

    run(examples_artificial, results_dir=Path('results_artificial'))
    run(examples_dry, results_dir=Path('results_dry'))

    mixlearn_dir = Path('results_mix_learn')
    if args['big_machine']:
        run([examples_mix_learn[0]], results_dir=mixlearn_dir, gpu=1)
        run([examples_mix_learn[1]], results_dir=mixlearn_dir, gpu=2)
    else:
        run(examples_mix_learn, results_dir=mixlearn_dir)

    gpu_pool.launch()
    print('the end')
