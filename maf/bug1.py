from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf


def run():
    print(tf.constant([1, 2, 3]))  # this line will crash!
    return "asd"


if __name__ == '__main__':
    print(tf.exp(2.0))
    executor: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=1)
    fut = executor.submit(run)
    t = fut.result()
    print(t)
