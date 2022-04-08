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
    if (type(P) is not np.ndarray or len(P.shape) != 2):
        return None

    n = P.shape[0]
    diag = np.diag(P)

    if (not np.any(diag == 1)):
        return False

    if (P == np.eye(n)).all():
        return True

    abs_s_idxs = np.where(diag == 1)[0]
    n_abss = 0

    for idx in abs_s_idxs:

        while(P[n_abss, n_abss] == 1):
            n_abss += 1

        if(idx > n_abss):

            permutation = []
            for i in range(n):
                if (i == n_abss):
                    permutation.append(idx)
                    permutation.append(i)
                elif (i != idx):
                    permutation.append(i)

            P[:] = P[permutation, :]
            P[:] = P[:, permutation]

            n_abss += 1

    n_abss = len(abs_s_idxs)
    R = P[n_abss:, :n_abss]
    Q = P[n_abss:, n_abss:]

    QminusI = np.eye(n - n_abss) - Q

    if np.linalg.det(QminusI) == 0:
        return False

    F = np.linalg.inv(QminusI)

    FR = np.matmul(F, R)

    for i in range(n_abss):
        if (not np.allclose(FR.sum(axis=1), np.ones((FR.shape[0], )))):
            return False

    return True
