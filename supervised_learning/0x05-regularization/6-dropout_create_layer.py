#!/usr/bin/env python3
"""Create a Layer with dropout"""


import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that creates a layer of a neural network using dropout:
    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function that should be used on the layer
    keep_prob is the probability that a node will be kept
    Returns: the output of the new layer"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    new_layer = tf.layers.Dense(n, activation=activation,
                                kernel_initializer=init)(prev)
    dropout_layer = tf.layers.Dropout(keep_prob)(new_layer)
    return dropout_layer
