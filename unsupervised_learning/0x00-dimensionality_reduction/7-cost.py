#!/usr/bin/env python3
""" Do the Dimensionality Reduction """
import numpy as np


def cost(P, Q):
    """
    calculates the cost of two matrices
    Returns:
        the cost
    """
    cerr = np.array([[1e-12]])

    return np.sum(P * np.log(np.maximum(P / np.maximum(Q, cerr), cerr)))
