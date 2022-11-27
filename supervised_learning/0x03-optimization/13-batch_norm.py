#!/usr/bin/env python3
"""Batch normalization"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output of a neural network using
    batch normalization:
    Z: is a numpy.ndarray of shape (m, n) that should be normalized
    m: is the number of data points
    n: is the number of features in Z
    gamma: is a numpy.ndarray of shape (1, n) containing the scales used for
    batch normalization
    beta: is a numpy.ndarray of shape (1, n) containing the offsets used for
    batch normalization
    epsilon: is a small number used to avoid division by zero
    Returns: the normalized Z matrix"""
    m = Z.mean(0)
    variance = Z.std(0) ** 2
    z_normalized = (Z - m) / ((variance + epsilon) ** 0.5)
    z_normalized = z_normalized * gamma + beta
    return z_normalized
