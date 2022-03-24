#!/usr/bin/env python3
"""8. t-SNE"""


import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """A function that performs a t-SNE transformation"""
    # X is a numpy.ndarray of shape (n, d) containing the dataset
    X = pca(X, idims)
    n, _ = X.shape
    # P is a numpy.ndarray of shape (n, n) containing the pairwise affinities
    P = P_affinities(X, perplexity=perplexity) * 4
    # Y is a numpy.ndarray of shape (n, ndims) containing the embedded points
    Y = np.random.randn(n, ndims)
    # iY is a numpy.ndarray of shape (n, ndims) containing the embedded points
    emc = np.zeros((n, ndims))

    for i in range(iterations):
        # Perform early exaggeration
        dY, Q = grads(Y, P)

        # momentum is the learning rate for the first 20 iterations
        if i <= 20:
            momentum = 0.5
        else:
            momentum = 0.8

        # Update the embedding
        Y = Y + (momentum * emc - lr * dY) - np.tile(np.mean(Y, 0), (n, 1))

        # Update the embedding for the next iteration
        if (i + 1) != 0 and (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        if (i + 1) == 100:
            P = P / 4.

    return Y
