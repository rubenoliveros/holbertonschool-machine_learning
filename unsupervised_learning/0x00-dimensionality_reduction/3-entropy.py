#!/usr/bin/env python3
"""3. Entropy"""


import numpy as np


def HP(Di, beta):
    """
    A function that calculates the Shannon entropy
    and P affinities relative to a data point
    """
    prob = np.exp(-Di * beta)
    total = np.sum(prob)
    Pi = prob / total
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
