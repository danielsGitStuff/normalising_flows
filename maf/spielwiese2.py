import os
from pathlib import Path

import numpy as np

from maf.DL import DL2

dir = Path("/home/xor/Documents/bepy/maf/mixlearn/.cache/mixlearn_synth_unconditional3/")
for d in os.listdir(dir):
    d = Path(dir, d)
    if not d.is_dir():
        continue
    if not DL2.can_load(d):
        continue
    dl = DL2.load(d)
    all = dl.get_conditional()
    print(d)
