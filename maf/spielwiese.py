from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt


def hist(df: pd.DataFrame, f: str):
    def h(df: pd.DataFrame, f: str):
        fig = plt.figure(figsize=(20, 20))
        ax = fig.gca()
        df.hist(ax=ax)
        fig.tight_layout()
        plt.savefig(f)
        plt.clf()
    d1 = df.loc[df[0] > 0]
    d0 = df.loc[df[0] < 1]
    h(d1, f"{f}_1.png")
    h(d0, f"{f}_0.png")


ds_samples = tf.data.experimental.load('/home/xor/Documents/bepy/maf/mixlearn/.cache/ds_synth4/signal/')
df_samples = pd.DataFrame(ds_samples.as_numpy_iterator())
desc_samples = df_samples.describe()
desc_samples.to_csv('spielwiese_samples.csv')

hist(df_samples, 'spielwiese_hist_samples')

print('that were samples')

ds_original = tf.data.experimental.load('/home/xor/Documents/bepy/maf/mixlearn/.cache/mixlearn_synth_unconditional3/ds_training')
df_original = pd.DataFrame(ds_original.as_numpy_iterator())
desc_original = df_original.describe()
desc_original.to_csv('spielwiese_original.csv')

df_original.hist()
hist(df_original, 'spielwiese_hist_original')

print('done')
