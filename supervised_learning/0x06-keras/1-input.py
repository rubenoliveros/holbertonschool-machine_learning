#!/usr/bin/env python3
"""1-Input"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that build a keras model
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the
    network
    activations is a list containing the activation functions used for each
    layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    Return: keras model"""
    regularizer = K.regularizers.L2(l2=lambtha)
    inputLayer = K.Input(shape=(nx,), name='input_1')
    lyr = K.layers.Dense(
        layers[0], activation=activations[0],
        kernel_regularizer=regularizer)(inputLayer)
    for i in range(1, len(layers)):
        dropoutLayer = K.layers.Dropout(1 - keep_prob)(lyr)
        lyr = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=regularizer)(dropoutLayer)
    return K.Model(inputLayer, lyr)
