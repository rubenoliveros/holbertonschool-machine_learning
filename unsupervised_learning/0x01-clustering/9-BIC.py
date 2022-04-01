#!/usr/bin/env python3
"""9. BIC"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    this function performs the Bayesian Information Criterion
    args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        kmin: is a positive integer containing the minimum number of clusters
        kmax: is a positive integer containing the maximum number of clusters
        iterations: is a positive integer containing the maximum number of
        iterations for the EM algorithm
        tol: is a non-negative float containing the tolerance for the EM
        algorithm
        verbose: is a boolean that determines if the EM algorithm should print
        information
    return:
        best_k: is the integer that resulted in the highest BIC value
        best_BIC: is the BIC value that resulted in the highest BIC value
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    pis = []
    msL = []
    salls = []
    lkhds = []
    bsall = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, lkhd = expectation_maximization(
            X,
            k,
            iterations,
            tol,
            verbose
        )
        pis.append(pi)
        msL.append(m)
        salls.append(S)
        lkhds.append(lkhd)

        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)

        bsall.append(p * np.log(n) - 2 * lkhd)

    lkhds = np.array(lkhds)
    bsall = np.array(bsall)
    best_k = np.argmin(bsall)
    best_result = (pis[best_k], msL[best_k], salls[best_k])

    return best_k + 1, best_result, lkhds, bsall
