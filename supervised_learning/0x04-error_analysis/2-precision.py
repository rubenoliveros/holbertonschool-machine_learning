#!/usr/bin/env python3
"""2. Precision"""
import numpy as np


def precision(confusion):
    """A function that calculates the precision in a confusion matrix"""
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
