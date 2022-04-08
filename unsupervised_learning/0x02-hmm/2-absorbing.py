#!/usr/bin/env python3
"""2. Absorbing Chains"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    Args:
       P is a is a square 2D numpy.ndarray of shape (n, n) representing the
         standard transition matrix
           - P[i, j] is the probability of transitioning from state i to
                     state j
           - n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """    
    if not isinstance(P, np.ndarray)\
            or len(P.shape) != 2\
            or P.shape[0] != P.shape[1]\
            or P.shape[0] < 1:
        return None

    if np.all(np.diag(P) == 1):
        return True

    if P[0, 0] != 1:
        return False

    P = P[1:, 1:]

    if np.all(np.count_nonzero(P, axis=0) > 2):
        return True
    else:
        return False
