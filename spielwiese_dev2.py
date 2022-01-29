import tensorflow as tf
import numpy as np
from tensorflow_probability.python.distributions import MultivariateNormalTriL

from common import jsonloader
from distributions.GaussianMultivariateFullCov import GaussianMultivariateFullCov

if __name__ == '__main__':
    from scipy.stats import multivariate_normal as M

    seed = 43
    tf.random.set_seed(seed)
    np.random.seed(seed)

    cov = [[1, .1], [.1, 1]]
    cov = np.array(cov, dtype=np.float32)
    loc = [0, 5]
    loc = np.array(loc, dtype=np.float32)
    samples_sc = M.rvs(mean=loc, covd=cov, size=10000)

    print(samples_sc[:10])

    g = GaussianMultivariateFullCov(loc=loc, cov=cov)
    samples_tfp = g.sample(10000)
    print(cov)
    print('dada')
    print(np.cov(samples_sc.T))
    print(np.cov(samples_tfp.T))
    print('SSAAMMPPLLEESS')
    print(samples_sc[:10])
    print('_____')
    print(samples_tfp[:10])

# if __name__ == '__main__':
#     # '`MultivariateNormalTriL(loc=loc, '
#     # 'scale_tril=tf.linalg.cholesky(covariance_matrix))` instead.',
#     cov = [[1.0, 2.0], [0.0, 1.0]]
#     d: MultivariateNormalTriL = MultivariateNormalTriL(scale_tril=tf.linalg.cholesky(cov))
#     wrapped = GaussianMultivariateFullCov(cov)
#     samples = wrapped.sample(3)
#     print(samples)
#     js = jsonloader.to_json(wrapped)
#     des: GaussianMultivariateFullCov = jsonloader.from_json(js)
#     samples = des.sample(3)
#     print(samples)
