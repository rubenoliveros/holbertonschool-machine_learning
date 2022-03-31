#!/usr/bin/env python3
"""5. PDF"""

import numpy as np


def pdf(X, m, S):
    """
    Function that calculates the probability
    args:
        - X: is a numpy.ndarray of shape (n, d) containing the data
            points whose probability density should be calculated
        - m: is a numpy.ndarray of shape (d,) containing the mean of
            the distribution
        - S: is a numpy.ndarray of shape (d, d) containing
            the covariance of the distribution
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    _, d = X.shape

    front = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(S))
    prop_mul = np.matmul((-(X - m) / 2), np.linalg.inv(S))
    dia_pe = np.matmul(prop_mul, (X - m).T).diagonal()

    pdf = np.exp(dia_pe) * front

    return np.where(pdf < 1e-300, 1e-300, pdf)
