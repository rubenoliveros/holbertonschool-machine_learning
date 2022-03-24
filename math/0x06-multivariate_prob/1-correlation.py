#!/usr/bin/env python3
"""1. Correlation"""


import numpy as np


def correlation(C):
    """A function that calculates a correlation matrix"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    d, d_2 = C.shape
    if d != d_2:
        raise ValueError("C must be a 2D square matrix")
    D = np.sqrt(np.diag(C))
    D_inverse = 1 / np.outer(D, D)
    corr = D_inverse * C
    return corr
