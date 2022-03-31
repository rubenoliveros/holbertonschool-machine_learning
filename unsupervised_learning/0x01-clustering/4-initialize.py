#!/usr/bin/env python3
"""4. Initialize GMM"""


import numpy as np


kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    function that initializes the centroids for K-means
    args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset that will
        be used for K-means
        k: is a positive integer containing the number of clusters
    Returns:
        C: is a numpy.ndarray of shape (k, d) containing the initialized
        centroids for K-means
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None

    c_mgen, _ = kmeans(X, k)
    # calc distance between the point i and the centroid j
    S = np.tile(np.diag(np.ones(d)), (k, 1)).reshape(k, d, d)

    return np.full(shape=(k,), fill_value=1/k), c_mgen, S
