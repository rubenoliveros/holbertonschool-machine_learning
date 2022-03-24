#!/usr/bin/env python3
"""6. Gradient"""


import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """A function that calculates the gradients of Y"""
    a, dim = Y.shape
    # q is the Q affinities of Y
    # num is the number of points in Y
    affinQ, num = Q_affinities(Y)

    # dY is the gradient of Y filled by zeros
    dY = np.zeros((a, dim))
    # b is the gradient of the Q affinities
    PQ = P - affinQ

    for i in range(a):
        # here we calculate the gradient of Y for
        # each point i in Y and add it dY
        dY[i, :] = np.sum(
            np.tile(PQ[:, i] * num[:, i], (dim, 1)).T * (Y[i, :] - Y),
            0
        )

    return dY, affinQ
