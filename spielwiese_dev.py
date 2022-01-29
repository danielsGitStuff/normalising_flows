import numpy as np
from tensorflow_probability.python.distributions import MultivariateNormalTriL

from common import jsonloader

if __name__ == '__main__':
    from scipy.stats import multivariate_normal as M

    seed = 43
    np.random.seed(seed)

    cov = [[1, .1], [.1, 1]]
    cov = np.array(cov, dtype=np.float32)
    # m = np.array([[1, 1, 1],
    #                 [1, 2, 1],
    #                 [1, 3, 2],
    #                 [1, 4, 3]])
    # cov = np.cov(m.T)
    print(cov)
    print('')
    loc = np.zeros(cov.shape[0])
    loc = np.array(loc, dtype=np.float32)
    samples_sc = M.rvs(mean=loc, cov=cov, size=10000)

    # samples_sc = np.array([[1, 2], [-1, -2]], dtype=np.float32)
    print(samples_sc)
    print('____cov____')
    print(np.cov(samples_sc.T))

    # v = np.var([1, -1])
    # print(v)
