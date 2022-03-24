#!/usr/bin/env python3
"""2. Initialize t-SNE"""


import numpy as np


def P_init(X, perplexity):
    """
    A function that initializes all variables
    required to calculate the P affinities in t-SNE
    """
    n = X.shape[0]
    mult = np.matmul(X, -X.T)
    summation = np.sum(np.square(X), 1)
    D = np.add(np.add(2 * mult, summation), summation.T)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return (D, P, betas, H)
