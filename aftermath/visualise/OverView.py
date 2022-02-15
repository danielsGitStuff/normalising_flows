from aftermath.Constants import Constants
from distributions.base import enable_memory_growth
from maf.dim2.NF2D_10Bumps import NF2D_10Bumps
from maf.dim2.NF2D_1Bumps import NF2D_1Bumps
from maf.dim2.NF2D_1Rect import NF2D_1Rect
from maf.dim2.NF2D_2Bumps import NF2D_2Bumps
from maf.dim2.NF2D_3Rect import NF2D_3Rect
from maf.dim2.NF2D_4Rect import NF2D_4Rect
from maf.dim2.NF2D_Diag4 import NF2D_Diag4
from maf.dim2.NF2D_RandomA import NF2D_RandomA
from maf.dim2.NF2D_RandomB import NF2D_RandomB
from maf.dim2.NF2D_Row3 import NF2D_Row3
from maf.dim2.NF2D_Row4 import NF2D_Row4
from maf.dim2.VisualExample2D import VisualExample2D
from typing import List, Type

from maf.stuff.MafExperiment import MafExperiment


class OverView(MafExperiment):
    def __init__(self):
        super().__init__('bogus name', pool_size=1)

    def _run(self):
        pass

    def print_divergences(self):
        pass

    def run(self):
        experiment_classes: List[Type[VisualExample2D]] = [NF2D_1Bumps, NF2D_1Rect, NF2D_2Bumps, NF2D_3Rect, NF2D_4Rect, NF2D_10Bumps, NF2D_Diag4, NF2D_RandomA, NF2D_RandomB,
                                                           NF2D_Row3,
                                                           NF2D_Row4]
        experiments: List[VisualExample2D] = []
        for ex_class in experiment_classes:
            experiments.append(ex_class())
        for ex in experiments:
            name = type(ex).__name__
            title = Constants.get_name_dict().get(name, "=== NOT IN DICT ===")
            print(f"init class '{name}' with title '{title}'")
        enable_memory_growth()
        for ex in experiments:
            name = type(ex).__name__
            title = Constants.get_name_dict().get(name, ex.create_data_title())
            print(f"heatmap of '{type(ex).__name__}' with title '{title}'")
            self.hm(dist=ex.data_distribution, title=title, xmin=ex.xmin, xmax=ex.xmax, ymin=ex.ymin, ymax=ex.ymax, mesh_count=ex.mesh_count)
            print('done')
        self.print_denses(name='test')


if __name__ == '__main__':
    OverView().run()
    print('exit')
