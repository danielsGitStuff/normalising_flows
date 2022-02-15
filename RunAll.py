from __future__ import annotations

import numpy as np

from maf.dim10.Dim10aCenteredMVG import Dim10aCenteredMVG
from maf.dim2.NF2D_RandomA import NF2D_RandomA
from maf.dim2.NF2D_RandomB import NF2D_RandomB
from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunnerBalanced import MixLearnExperimentMiniBooneClfVarRunnerBalanced
from maf.dim1.NF1D_1Bumps import NF1D_1Bumps
from maf.dim1.NF1D_2Bumps import NF1D_2Bumps
from maf.dim2.NF2D_10Bumps import NF2D_10Bumps
from maf.dim2.NF2D_1Bumps import NF2D_1Bumps
from maf.dim2.NF2D_1Rect import NF2D_1Rect
from maf.dim2.NF2D_2Bumps import NF2D_2Bumps
from maf.dim2.NF2D_3Rect import NF2D_3Rect
from maf.dim2.NF2D_4Rect import NF2D_4Rect
from maf.dim2.NF2D_Diag4 import NF2D_Diag4
from maf.dim2.NF2D_Row3 import NF2D_Row3
from maf.dim2.SS1DExample import SS1DMafExperiment
from maf.dim2.SS2DExample import SS2DMafExperiment
from maf.dim2.ShowCase1D1 import ShowCase1D1
from pathlib import Path

from RunAllProcessWrapper import GPUProcessWrapper, GPUProcessWrapperPool
from keta.argparseer import ArgParser
from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner import MixLearnExperimentMiniBooneClfVarRunner

import random
from typing import List, Type, Dict, Any

from maf.dim2.NF2D_Row4 import NF2D_Row4
from maf.dim10.Dim10bLargeGaps import Dim10bLargeGaps
from maf.dim10.Dim10cSmallGaps import Dim10cSmallGaps

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
                                       NF2D_Row4,
                                       NF2D_RandomA,
                                       NF2D_RandomB
                                       ]

    examples_dry: List[Type] = [Dim10aCenteredMVG,
                                Dim10bLargeGaps,
                                Dim10cSmallGaps
                                ]

    # this speeds up training!
    # Global.Testing.set('testing_epochs', 1)
    # Global.Testing.set('testing_nf_layers', 1)
    # Global.Testing.set('testing_nf_norm_layer', False)

    examples_mix_learn: List[Type] = [MixLearnExperimentMiniBooneClfVarRunner,
                                      MixLearnExperimentMiniBooneClfVarRunnerBalanced]

    # make sure the dataset is already in place before starting processes relying on it. Might cause race conditions otherwise
    # test_pool = GPUProcessWrapperPool()
    # pw = GPUProcessWrapper(module=MinibooneDL3.__module__, klass=MinibooneDL3.__name__, results_dir='nanana useless')
    # pw.execute()

    to_run = examples_artificial + examples_dry + examples_mix_learn
    random.shuffle(to_run)
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    to_runs = np.array_split(to_run, len(gpus))
    for i, to_run in enumerate(to_runs):
        run(to_run, results_dir=None, gpu=i)

    run([Dim10bLargeGaps], results_dir=Path('results_artificial_dim10'))
    run([Dim10cSmallGaps], results_dir=Path('results_artificial_dim10'))

    gpu_pool.launch()
    print('the end')
