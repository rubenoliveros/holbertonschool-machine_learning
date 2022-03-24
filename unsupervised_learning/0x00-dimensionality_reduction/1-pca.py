#!/usr/bin/env python3
"""1. PCA v2"""


import numpy as np


def pca(X, ndim):
    """A function that performs PCA on a dataset"""
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    u, s, v = np.linalg.svd(A)
    W = v.T[:, :ndim]
    T = np.matmul(A, W)
    return (T)
