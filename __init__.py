import sys

from pathlib import Path

root: Path = Path(__file__).parent
print(f"--- {root.absolute()}")
# sys.path.append(str(root.absolute()))

from . import maf
from . import common
from .common import util, jsonloader

# from maf.MaskedAutoregressiveFlow import MaskedAutoregressiveFlow
# from distributions.base import enable_memory_growth, BaseMethods, TData, TDataOpt
# from distributions.LearnedDistribution import LearnedConfig, LearnedDistribution
# import normalising_flows.common.jsonloader as jsonloader
# import normalising_flows.common.util as util

