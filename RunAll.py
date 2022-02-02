from __future__ import annotations

from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunnerBalanced import MixLearnExperimentMiniBooneClfVarRunnerBalanced
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

from RunAllProcessWrapper import ProcessWrapper
from keta.argparseer import ArgParser
from maf.dry_examples.EvalExample3 import EvalExample3
from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner import MixLearnExperimentMiniBooneClfVarRunner

from common.globals import Global
from typing import List, Type

from maf.visual_examples.NF2D_Row4 import NF2D_Row4

if __name__ == '__main__':
    ArgParser.parse()
    description: str = f"""
    This script runs all MAF experiments inlcuding learning artificial distributions like multivariate/multimodal Gaussians, Miniboone, MNIST and experiments to show divergence.

    """
    print(description)


    def run(examples: List[Type]):
        for t in examples:
            pw = ProcessWrapper(module=t.__module__, klass=t.__name__)
            pw.execute()


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
                                       NF2D_Row4]

    examples_dry: List[Type] = [  # EvalExample,
        # EvalExample2,
        EvalExample3,
        # EvalExample4
    ]

    Global.Testing.set('testing_epochs', 1)
    Global.Testing.set('testing_nf_layers', 1)
    Global.Testing.set('testing_nf_norm_layer', False)

    examples_mix_learn: List[Type] = [MixLearnExperimentMiniBooneClfVarRunner,
                                      MixLearnExperimentMiniBooneClfVarRunnerBalanced]

    Global.set_global('results_dir', Path('results_artificial'))
    run(examples_artificial)
    Global.set_global('results_dir', Path('results_dry'))
    run(examples_dry)
    # Global.set_global('results_dir', Path('results_mix_learn'))
    # run(examples_mix_learn)
