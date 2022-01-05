#!/usr/bin/env python3
"""24. One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """a function that converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None
    A = np.eye(classes)[Y]
    return A.T
