import hashlib
import io
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import strptime, mktime
from typing import TYPE_CHECKING, List, Tuple, Set, Any, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.utils import shuffle
if TYPE_CHECKING:
    pass

# np.random.seed(1234)
secs = int(datetime.now().timestamp())
np.random.seed(secs)




class Runtime:
    def __init__(self, name: str):
        self.name: str = name
        self._start: datetime.date = None
        self._summed: timedelta = None

    def start(self):
        self._start: datetime.date = datetime.now()
        return self

    def stop(self):
        delta: timedelta = datetime.now() - self._start
        if self._summed is None:
            self._summed: timedelta = delta
        else:
            self._summed: timedelta = self._summed + delta
        return self

    def reset(self):
        self._start = None
        self._summed = None
        return self

    def print(self):
        s = f"Runtime for '{self.name}' is {self._summed}"
        pri(s)
        return self

    def to_string(self):
        return f"r({self.name})={self._summed}"


def debug_print_df(df: DataFrame, f: [str, Path]):
    drop_keys = [k for k in df.keys() if k.startswith("val_") and k != "val_reward/e"]
    drop_keys += ["time", "run", "epoch"]
    d: DataFrame = df.loc[df.epoch > 100].drop(drop_keys, axis=1)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 10))
    sns.heatmap(d.corr(), xticklabels=True, yticklabels=True, annot=True, cmap="coolwarm")
    plt.savefig(f)
    plt.clf()


def merge_sets(*sets: List[Set[Any]]) -> Set[Any]:
    result = set()
    [result.add(v) for s in sets for v in s]
    return result


def randomSeed():
    seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder) / 2
    seed += int.from_bytes(os.getrandom(4), byteorder=sys.byteorder) / 2
    seed = int(seed)
    return seed


def right_strip(s: str, suffix: str) -> str:
    if s.endswith(suffix):
        return s[:len(s) - len(suffix)]
    return s


def left_strip(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def print_confusion_matrix(matrix, title, sub_title=None):
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="YlGn")

    cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
    # cbar.ax.set_ylabel("debuglabel", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(["Pred=A", "Pred=B", "Pred=D"])
    ax.set_yticklabels(["Exp=A", "Exp=B", "Exp=D"])
    original_title = title
    if sub_title is not None:
        title = "{} [{}]".format(title, sub_title)
    ax.set(title=title)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    textcolors = ["black", "white"]
    threshold = im.norm(matrix.max()) / 2.
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=14)
    for i in range(3):
        for j in range(3):
            color = textcolors[int(im.norm(matrix[i, j]) > threshold)]
            u = valfmt(matrix[i, j], None)
            text = im.axes.text(j, i, u, horizontalalignment="center",
                                verticalalignment="center", fontsize=24, color=color)

    plt.show()
    return fig, original_title


def normalize_seasons(seasons):
    season_count = len(seasons)
    scale = 1
    normalized_seasons = {season: 1 / season_count * i + 1 / season_count * scale for (i, season) in
                          enumerate(seasons)}
    return normalized_seasons


def read_text_file(path):
    f = io.open(path, mode="r", encoding="utf-8")
    text = f.read()
    f.close()
    return text


def dmy(s):
    date = strptime(s, "%d.%m.%Y")
    date = datetime.fromtimestamp(mktime(date)).date()
    return date


def exp_win(prob, quote):
    if prob is None or quote is None:
        eprint("debug error 123321")
    return (-1) * (1 - prob) + prob * quote


def is_later_than(before, after):
    # todo check if bugg
    eprint("Possible bug: this has not been checked (str vs date comparison)")
    if isinstance(before, str):
        before = dmy(before)
    if isinstance(after, str):
        after = dmy(after)
    return after > before


def sort_after_xs(xs, ys):
    s = sorted(zip(xs, ys))
    sorted_xs = [x for x, y in s]
    sorted_ys = [y for x, y in s]
    return sorted_xs, sorted_ys


def write_text_file(path, text):
    f = io.open(path, mode="w", encoding="utf-8")
    f.write(text)
    f.close()


def parse_int_pair(string):
    try:
        load_seasons = re.split(",", string)
        start = int(load_seasons[0])
        stop = int(load_seasons[1])
        pair_list = list(range(start, stop + 1))
    except:
        print("could not parse seasons!")
        raise AttributeError("could not parse '{}' as a pair of ints".format(string))
    return pair_list


def parse_int_triple(string):
    try:
        load_seasons = re.split(",", string)
        a = int(load_seasons[0])
        b = int(load_seasons[1])
        c = int(load_seasons[2])
        triple = (a, b, c)
    except:
        print("could not parse triple!")
        raise AttributeError("could not parse '{}' as a triple of ints".format(string))
    return triple


def eprint(*args, **kwargs):
    pri(s=str(args), error=True)


def pri(s: str, error: bool = False):
    d = datetime.now()
    line = "[{:02d}:{:02d}:{:02d}]: ".format(d.hour, d.minute, d.second) + s
    if error:
        print(line, file=sys.stderr)
    else:
        print(line)


def ymd(s) -> datetime.date:
    if s is None:
        return None
    # if isinstance(s, datetime.date):
    #     return s
    date = strptime(s, "%Y-%m-%d")
    date = datetime.fromtimestamp(mktime(date)).date()
    return date


def ymd_hms(s) -> datetime:
    if s is None:
        return None
    # if isinstance(s, datetime.date):
    #     return s
    date = strptime(s, "%Y-%m-%d %H:%M:%S")
    date = datetime.fromtimestamp(mktime(date))
    return date


def ymd_hmsf(s: str) -> datetime:
    if s is None:
        return None
    # if isinstance(s, datetime.date):
    #     return s
    date = strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f")
    date = datetime.fromtimestamp(mktime(date))
    return date


def md5_dicts_and_lists(*args):
    m = hashlib.md5()
    for obj in args:
        if isinstance(obj, dict):
            for (k, v) in obj.items():
                m.update(str(k).encode("utf-8"))
                m.update(str(v).encode("utf-8"))
        elif isinstance(obj, list):
            for v in obj:
                m.update(str(v).encode("utf-8"))
    return m.hexdigest()


def parse_date(date_line: str, param: str):
    date = strptime(date_line, param)
    d = datetime.fromtimestamp(mktime(date)).date()
    return d


def parse_date_time(date_line):
    date = strptime(date_line)
    d = datetime.fromtimestamp(mktime(date))
    return d


def set_seed(seed_value: int):
    for _ in range(1):
        print(f"FIXING SEED TO {seed_value}", file=sys.stderr)
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    # import os
    # os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    # from keras import backend as K
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)
    # for later versions:
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)


class Util:
    __print_lines__ = []


def p(*args):
    Util.__print_lines__.append(str(*args))
    print(*args)
    # print(str(*args))


def reprint():
    [print(s) for s in Util.__print_lines__]
    __print_lines__ = []


def inverse_map(m: map):
    return {v: k for k, v in m.items()}


def ys_to_win(ys):
    assert len(ys) == 3
    num = sum([y * 2 ** i for i, y in enumerate(ys)])
    if num == 1:
        return "A"
    elif num == 2:
        return "B"
    elif num == 4:
        return "D"
    else:
        raise AttributeError("only one outcome is possible")


def toss_coin(bias=0.5):
    """:parameter bias probability of returning True"""
    if np.random.binomial(1, bias) == 1:
        return True
    else:
        return False


def cp(source: str, target: str, overWrite=True):
    if not overWrite and os.path.exists(target):
        return
    p("copying '{}' -> '{}'".format(source, target))
    shutil.copy(src=source, dst=target)


def calcModelName(foretellSeason: int, targetSeason: int, trainingSeason0: int, trainingSeasonN: int, sliceIndex: int = None):
    name = "foretell_{}.t_{}.t0_{}.tn_{}".format(foretellSeason, targetSeason, trainingSeason0, trainingSeasonN)
    if sliceIndex is not None:
        name = "{}.si_{}".format(name, sliceIndex)
    return name


def printDirectoryMd5s(directory: str) -> str:
    r = subprocess.run("md5sum {}{}*".format(directory, os.sep), shell=True, capture_output=True)
    s = r.stdout.decode("utf-8")
    print("##### MD5 sums of '{}' #####".format(directory))
    print(s)
    print("##### END #####")
    return s


def shuffle_lists(a: List, b: List) -> Tuple[List, List]:
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def np_shuffle(lists: List[Union[np.ndarray, List]]) -> List[np.ndarray]:
    lists: List[np.ndarray] = [np.array(ls) for ls in lists]
    shuffled = shuffle(*lists)
    return shuffled


def np_shuffle_t(lists: List[Union[np.ndarray, List]]) -> Tuple[np.ndarray]:
    lists: List[np.ndarray] = [np.array(ls) for ls in lists]
    shuffled = shuffle(*lists)
    return tuple(shuffled)


class CounterString:
    def __init__(self, pre: str):
        self.__pre: str = pre
        self.__count = -1

    def get(self) -> str:
        self.__count += 1
        return f"{self.__pre}.{self.__count}"
