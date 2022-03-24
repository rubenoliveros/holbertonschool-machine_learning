#!/usr/bin/env python3
"""4. P affinities"""


import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    A function that calculates the symmetric P affinities of a data set
    """
    # P is initialized to the identity matrix
    No, _ = X.shape

    # Compute pairwise distances
    # H is the Shannon entropy of P
    # P is the symmetric normalized P affinities
    # D is the pairwise distances
    # beta is the precision of the Gaussian distribution used for
    D, P, bgh, H = P_init(X, perplexity)

    for i in range(No):
        ms_neight = np.ones(D[i].shape, dtype=bool)
        ms_neight[i] = 0

        Hi, P[i][ms_neight] = HP(D[i][ms_neight], bgh[i])

        high = None
        low = 0

        # tol is a tolerance for finding the perplexity
        while abs(Hi - H) > tol:
            # validate if Hi is a valid perplexity
            if Hi < H:
                high = bgh[i, 0]
                bgh[i, 0] = (high + low) / 2
            else:
                low = bgh[i, 0]
                if high is None:
                    bgh[i, 0] *= 2
                else:
                    bgh[i, 0] = (high + low) / 2

            Hi, P[i][ms_neight] = HP(D[i][ms_neight], bgh[i])

    return (P + P.T) / (2 * No)
