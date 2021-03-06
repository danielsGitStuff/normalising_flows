from common.globals import Global
from common.argparser import ArgParser
from pathlib import Path

import tensorflow as tf
import pandas as pd
from distributions.LearnedTransformedDistribution import LearnedTransformedDistribution
from distributions.base import enable_memory_growth
from maf.dim2.MNIST import Mist
from maf.stuff.StaticMethods import StaticMethods

if __name__ == '__main__':
    ArgParser.parse()
    tf.random.set_seed(42)
    enable_memory_growth()

    # tf.debugging.experimental.enable_dump_debug_info(
    #     "tfdebug/",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1)
    #
    # Set CPU as available physical device
    # import tensorflow as tf
    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
    cache_dir = StaticMethods.cache_dir()
    checkpoints_dir = Path(cache_dir, "mnist_checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    prefix = "mnist_test"
    result_dir: Path = Path("results_mnist")
    Global.set_global('results_dir', result_dir)
    mist = Mist(conditional=True,
                one_hot=True,
                numbers=[2, 3, 4, 5],
                norm_layer=False,
                norm_data='-1 1',
                dataset_noise_variance=0.0,
                noise_layer_variance=0.1,
                batch_norm=True,
                epochs=150,
                use_tanh_made=True,
                layers=30,
                hidden_shape=[1024, 1024])
    if LearnedTransformedDistribution.can_load_from(folder=checkpoints_dir, prefix=prefix):
        mist.maf = LearnedTransformedDistribution.load(folder=checkpoints_dir, prefix=prefix)
        if mist.conditional != mist.maf.conditional:
            raise RuntimeError(f"conditional required: {mist.conditional}. loaded from disk: {mist.maf.conditional}")
    else:
        mist.fit()
        mist.maf.save(folder=checkpoints_dir, prefix=prefix)
        hdf = pd.DataFrame(mist.maf.history.to_dict())
        history_file = Path(checkpoints_dir, f"{prefix}_history.csv")
        hdf.to_csv(history_file)
        # hdf.to_csv(f"checkpoints{os.sep}{prefix}_history.csv")
    print('will sample and print results now')
    mist.test()
    print("end")
