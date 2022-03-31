#!/usr/bin/env python3
"""2. Variance"""


import numpy as np


def variance(X, C):
    """
    the function that calculates the total intra-cluster variance for a
    a dataset
    args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        C: is a numpy.ndarray of shape (k, d) containing the centroids
            for the clusters
    Returns:
        variance: is the total variance of all clusters
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    n, d = X.shape
    k = C.shape[0]

    if k > n:
        return None

    if C.shape[1] != d:
        return None

    x_var = np.repeat(X, k, axis=0).reshape(n, k, d)
    cen_var = np.tile(C, (n, 1)).reshape(n, k, d)

    dist = np.linalg.norm(x_var - cen_var, axis=2)

    # short distance
    return np.sum(np.min(dist ** 2, axis=1))
