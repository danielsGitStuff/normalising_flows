from pathlib import Path

from distributions.base import cast_dataset_to_tensor
from maf.DS import DS


def show_means(ds: DS):
    t, _ = cast_dataset_to_tensor(ds)
    import tensorflow as tf
    means = tf.reduce_mean(t, axis=0)
    print(f"means are: {means}")


def show_binary_class_distribution(ds: DS, path=''):
    t, _ = cast_dataset_to_tensor(ds)
    import tensorflow as tf
    mean = tf.reduce_mean(t[:, 0])
    print(f"binary class ration (0 or 1) is {mean} of {len(t)} samples in '{path}'")


def load_binary_class_distribution(path_to_data_set: [str, Path]):
    path = str(path_to_data_set)
    import tensorflow as tf
    ds = tf.data.experimental.load(path)
    show_binary_class_distribution(ds, path=path)


if __name__ == '__main__':
    load_binary_class_distribution('/home/xor/Documents/bepy/maf/mixlearn/.cache/mixlearn_mb_small/ds_training/')
    load_binary_class_distribution('/home/xor/Documents/bepy/maf/mixlearn/.cache/mixlearn_mb_small/ds_test/')
    print("########")
    load_binary_class_distribution('/home/xor/Documents/bepy/maf/mixlearn/.cache/mixlearn_mb_small/nf_116636_ds_training/')
    load_binary_class_distribution('/home/xor/Documents/bepy/maf/mixlearn/.cache/mixlearn_mb_small/nf_116636_ds_val/')
