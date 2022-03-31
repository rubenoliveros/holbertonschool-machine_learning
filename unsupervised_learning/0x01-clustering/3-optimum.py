#!/usr/bin/env python3
"""3. Optimize k"""


import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    the optimum number of clusters is the one that maximizes
    the variance of the clusters
    args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        kmin: is a positive integer containing the minimum number of
            clusters to check for (inclusive)
        kmax: is a positive integer containing the maximum number of
            clusters to check for (inclusive)
        iterations: is a positive integer containing the maximum
            number of iterations for K-means
    Returns:
        best_k: is the optimum number of clusters
        best_score: is the corresponding score
    """
    if (type(X) is not np.ndarray or len(X.shape) != 2):
        return None, None

    if (type(kmin) is not int or kmin < 1):
        return None, None

    if (kmax is None):
        kmax = X.shape[0]

    elif (type(kmax) is not int or kmax <= kmin):
        return None, None

    tup_res = []
    d_vars = []
    for k in range(kmin, kmax + 1):

        cent, clss = kmeans(X, k, iterations)
        if (cent is None or clss is None):
            return None, None

        var = variance(X, cent)
        if (var is None):
            return None, None

        var = float(var)

        if (k == kmin):
            first_var = var

        tup_res.append((cent, clss))
        d_vars.append(first_var - var)

    return tup_res, d_vars
