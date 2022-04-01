#!/usr/bin/env python3
"""7. Maximization"""

import numpy as np


def maximization(X, g):
    """
    maximize the log-likelihood of a GMM:
    args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        g: is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
    return (numpy.ndarray): the means for each cluster, or None on failure
    """
    if type(X) is not np.ndarray or type(g) is not np.ndarray:
        return (None, None, None)

    if len(X.shape) != 2 or len(g.shape) != 2:
        return (None, None, None)

    if X.shape[0] != g.shape[1]:
        return (None, None, None)

    if not np.isclose(np.sum(g, axis=0), 1).all():
        return (None, None, None)

    n, d = X.shape
    k, _ = g.shape

    m = np.dot(g, X) / g.sum(1)[:, None]
    covs = np.zeros([k, d, d])

    for i in range(k):
        ys = X - m[i, :]
        covs[i] = (
            g[i, :, None, None] *
            np.matmul(ys[:, :, None], ys[:, None, :])
        ).sum(axis=0)
    covs = covs / g.sum(axis=1)[:, None, None]

    return g.sum(axis=1) / n, m, covs
