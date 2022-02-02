from __future__ import annotations

from maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunnerBalanced import MixLearnExperimentMiniBooneClfVarRunnerBalanced
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
            print(f"executing example '{t.__name__}'")
            pw = ProcessWrapper(module=t.__module__, klass=t.__name__)
            pw.execute()
            print(f"example '{t.__name__}' done")


    examples_artificial: List[Type] = [  # NF1D_1Bumps,
        # NF1D_2Bumps,
        # NF2D_1Bumps,
        # NF2D_2Bumps,
        # NF2D_10Bumps,
        # SS1DMafExperiment,
        # SS2DMafExperiment,
        # ShowCase1D1,
        # NF2D_Diag4,
        # NF2D_Row3,
        NF2D_Row4]

    examples_dry: List[Type] = [  # EvalExample,
        # EvalExample2,
        EvalExample3,
        # EvalExample4
    ]

    Global.Testing.set('testing_epochs', 1)
    Global.Testing.set('testing_nf_layers', 1)

    examples_mix_learn: List[Type] = [MixLearnExperimentMiniBooneClfVarRunner,
                                      MixLearnExperimentMiniBooneClfVarRunnerBalanced]

    # Global.set_global('results_dir', Path('results_artificial'))
    # run(examples_artificial)
    # Global.set_global('results_dir', Path('results_dry'))
    # run(examples_dry)
    Global.set_global('results_dir', Path('results_mix_learn'))
    run(examples_mix_learn)
