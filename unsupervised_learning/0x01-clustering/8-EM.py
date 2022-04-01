#!/usr/bin/env python3
"""8. EM"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    this function performs expectation maximization on a dataset (X)
    args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        k: is the number of clusters
        iterations: is the maximum number of iterations to perform
        tol: is the tolerance of the EM algorithm
        verbose: is a boolean that determines if you should print information
        during the computation of the EM algorithm
    Returns:
        pi, m, S, g, or None, None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors
        for each cluster
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster
        g is a numpy.ndarray of shape (k, n) containing the probabilities for
            each data point in each cluster
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None

    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None, None

    prev = 0
    pi, m, S = initialize(X, k)
    g, log = expectation(X, pi, m, S)

    for i in range(iterations):
        if verbose and (i % 10 == 0):
            print(
                'Log Likelihood after {} iterations: {}'.format(
                    i,
                    log.round(5)  # round to 5 decimal places
                )
            )

        pi, m, S = maximization(X, g)
        g, log = expectation(X, pi, m, S)

        if abs(prev - log) <= tol:
            break

        prev = log

    if verbose:
        print(
            'Log Likelihood after {} iterations: {}'.format(
                i + 1,
                log.round(5)
            )
        )

    return pi, m, S, g, log
