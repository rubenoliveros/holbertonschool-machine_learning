#!/usr/bin/env python3
"""Dropout gradient descent"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network with Dropout
    regularization using gradient descent:
    Y: is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes: is the number of classes
        m: is the number of data points
    weights: is a dictionary of the weights and biases of the neural network
    cache: is a dictionary of the outputs and dropout masks of each layer of
    the neural network
    alpha: is the learning rate
    keep_prob: is the probability that a node will be kept
    L: is the number of layers of the network"""
    m = Y.shape[1]
    la = L
    a = 'A' + str(la)
    W = 'W' + str(la)
    b = 'b' + str(la)
    dz = cache[a] - Y
    dw = (np.dot(cache['A' + str(la - 1)], dz.T) / m).T
    db = np.sum(dz, axis=1, keepdims=True) / m
    weights[W] = weights[W] - alpha * dw
    weights[b] = weights[b] - alpha * db

    for la in range(L - 1, 0, -1):
        a = 'A' + str(la)
        W = 'W' + str(la)
        b = 'b' + str(la)
        wNext = 'W' + str(la + 1)
        aNext = 'A' + str(la - 1)
        g = (1 - cache[a]**2)
        dz = np.dot(weights[wNext].T,
                    dz) * g * cache['D' + str(la)] / keep_prob
        dw = (np.dot(cache[aNext], dz.T) / m).T
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights[W] = weights[W] - alpha * dw
        weights[b] = weights[b] - alpha * db
