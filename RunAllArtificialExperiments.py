from keta.argparseer import ArgParser
from pathlib import Path

from common.globals import Global
from typing import List

from maf.examples.stuff.MafExperiment import MafExperiment
from maf.examples.NF1D_1Bumps import NF1D_1Bumps
from maf.examples.NF1D_2Bumps import NF1D_2Bumps
from maf.examples.NF2D_10Bumps import NF2D_10Bumps
from maf.examples.NF2D_1Bumps import NF2D_1Bumps
from maf.examples.NF2D_2Bumps import NF2D_2Bumps
from maf.examples.NF2D_Diag4 import NF2D_Diag4
from maf.examples.NF2D_Row3 import NF2D_Row3
from maf.examples.NF2D_Row4 import NF2D_Row4
from maf.examples.SS1DExample import SS1DMafExperiment
from maf.examples.SS2DExample import SS2DMafExperiment
from maf.examples.ShowCase1D1 import ShowCase1D1

if __name__ == '__main__':
    ArgParser.parse()
    result_dir: Path = Path("results_artificial")
    Global.set_global('results_dir', result_dir)
    description: str = f"""
    This script runs all MAF experiments that learn artificial distributions like multivariate/multimodal Gaussians.
    Results go to '{result_dir.absolute()}' 
    """
    print(description)
    examples: List[MafExperiment] = [NF1D_1Bumps(), NF1D_2Bumps(), NF2D_1Bumps(), NF2D_2Bumps(), NF2D_10Bumps(), SS1DMafExperiment(), SS2DMafExperiment(), ShowCase1D1(),
                                     NF2D_Diag4(), NF2D_Row3(),
                                     NF2D_Row4()]
    for example in examples:
        print(f"executing example '{example.name}'")
        example.run()
        print(f"example '{example.name}' done")
