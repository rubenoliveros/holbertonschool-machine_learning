#!/usr/bin/env python3
"""1. Regular Chains"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    Args:
        P is a square 2D numpy.ndarray of shape (n, n) representing the
          transition matrix
            - P[i, j] is the probability of transitioning from state i to
                      state j
            - n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
             probabilities, or None on failure
    """
    if len(P.shape) != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return None

    P = np.linalg.matrix_power(P, 100)
    if np.any(P <= 0):
        return None
    ss_prob = np.array([P[0]])

    return ss_prob
