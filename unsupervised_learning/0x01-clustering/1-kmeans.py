#!/usr/bin/env python3
""" 1. K-mean"""

import numpy as np


def initialize(X, k):
    """
    This function initializes the centroids for K-means
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset that
        we want to cluster
        k: is a positive integer containing the number of clusters
    Returns:
        [type]: [description]
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None

    # initialize the centroids
    _, d = X.shape

    return np.random.uniform(
        np.amin(X, axis=0),
        np.amax(X, axis=0),
        (k, d)
    )


def kmeans(X, k, iterations=1000):
    """
    this function performs K-means on a dataset
    Arguments:
    - X is a numpy.ndarray of shape (n, d) containing the dataset
    - k is a positive integer containing the number of clusters
    - iterations is a positive integer containing the maximum number of
        iterations the algorithm should perform
    Returns:
        - C, or None on failure
    """

    # Initialize the cluster centroids
    cen = initialize(X, k)

    if cen is None:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # vectorization
    for _ in range(iterations):
        centroides_prev = np.copy(cen)

        # assign the cb
        x_vec = np.repeat(X, k, axis=0).reshape(n, k, d)
        cen_vec = np.tile(cen, (n, 1)).reshape(n, k, d)

        clss = np.argmin(
            np.linalg.norm(x_vec - cen_vec, axis=2) ** 2,
            axis=1
        )

        for j in range(k):
            indices = np.where(clss == j)[0]

            # check for empty clusters
            if len(indices) == 0:
                cen[j] = initialize(X, 1)
            else:
                cen[j] = np.mean(X[indices], axis=0)

        if (cen == centroides_prev).all():
            return cen, clss

    cen_vec = np.tile(cen, (n, 1))
    cen_vec = cen_vec.reshape(n, k, d)
    dist = np.linalg.norm(x_vec - cen_vec, axis=2)
    clss = np.argmin(dist ** 2, axis=1)

    return cen, clss
