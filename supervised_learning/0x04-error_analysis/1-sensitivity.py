#!/usr/bin/env python3
"""1. Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """A function that calculates the sensitivity for a confusion matrix"""
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
