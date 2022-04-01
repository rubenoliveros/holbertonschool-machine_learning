#!/usr/bin/env python3
"""6. Expectation"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    this function calculates the probability of a data point
    args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        pi: is a numpy.ndarray of shape (k, 1) containing the priors
        m: is a numpy.ndarray of shape (k, d) containing the means of
            the gaussian distributions
        S: is a numpy.ndarray of shape (k, d, d) containing the cov
            matrices of the gaussian distributions
    return:
        g: is a numpy.ndarray of shape (k, n) containing the probabilities
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape

    if pi.shape[0] > n:
        return None, None

    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None

    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    s_tg = np.sum(g, axis=0, keepdims=True)
    g /= s_tg

    return g, np.sum(np.log(s_tg))
