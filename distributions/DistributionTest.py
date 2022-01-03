from unittest import TestCase

from distributions.Distribution import Distribution
from distributions.GaussianMultivariate import GaussianMultivariate
from distributions.kl.KLSampler import KLSampler
from maf.DS import DS
import tensorflow as tf


class DistributionTest(TestCase):
    def test_kl_1d(self):
        g1: Distribution = GaussianMultivariate(input_dim=1, mus=[0], cov=[1])
        g2: Distribution = GaussianMultivariate(input_dim=1, mus=[0], cov=[1])
        kl = g1.kl_divergence(g2)
        kl_tf = g1.tfd_distribution.kl_divergence(g2.tfd_distribution)
        self.assertEqual(kl, kl_tf)

    def test_kl_2d(self):
        g1: Distribution = GaussianMultivariate(input_dim=2, mus=[0, 0], cov=[1, 1])
        g2: Distribution = GaussianMultivariate(input_dim=2, mus=[0, 0], cov=[2, 1])
        kl = g1.kl_divergence(g2)
        kl_tf = g1.tfd_distribution.kl_divergence(g2.tfd_distribution)
        self.assertEqual(kl, kl_tf)

    def test_i(self):
        def gen():
            ragged_tensor = tf.ragged.constant([[1, 2], [3]])
            yield 42, ragged_tensor

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))

        list(dataset.take(1))

        t = KLSampler(dims=3, half_width=1.0, step_size=.01, batch_size=1000)

        # ds = DS.from_generator(gen, output_signature=(tf.TensorSpec(shape=(None, 5), dtype=tf.float32)))
        # asdasd = (tf.TensorSpec(shape=(None, 5), dtype=tf.float32))
        ds = DS.from_generator(t.to_gen(), output_signature=t.get_output_signature())
        print("starting ...")
        for i in range(len(t.batch_sizes)):
            print(f"i {i}")
            if i == 1:
                print("ads")
            b = t.__getitem__(i)
            print(b)
        # for b in ds:
        #     print('batch')
        #     print(b)


