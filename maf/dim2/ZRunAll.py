from typing import List

from maf.stuff.MafExperiment import MafExperiment
from maf.dim2.NF1D_1Bumps import NF1D_1Bumps
from maf.dim2.NF1D_2Bumps import NF1D_2Bumps
from maf.dim2.NF2D_10Bumps import NF2D_10Bumps
from maf.dim2.NF2D_1Bumps import NF2D_1Bumps
from maf.dim2.NF2D_2Bumps import NF2D_2Bumps
from maf.dim2.NF2D_Diag4 import NF2D_Diag4
from maf.dim2.NF2D_Row3 import NF2D_Row3
from maf.dim2.NF2D_Row4 import NF2D_Row4
from maf.dim2.SS1DExample import SS1DMafExperiment
from maf.dim2.SS2DExample import SS2DMafExperiment
from maf.dim2.ShowCase1D1 import ShowCase1D1

if __name__ == '__main__':
    examples: List[MafExperiment] = [NF1D_1Bumps(), NF1D_2Bumps(), NF2D_1Bumps(), NF2D_2Bumps(), NF2D_10Bumps(), SS1DMafExperiment(), SS2DMafExperiment(), ShowCase1D1(), NF2D_Diag4(), NF2D_Row3(),
                                     NF2D_Row4()]
    for example in examples:
        print(f"executing example '{example.name}'")
        example.run()
        print(f"example '{example.name}' done")
