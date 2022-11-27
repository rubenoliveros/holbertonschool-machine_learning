#!/usr/bin/env python3
"""Batch normalization"""


import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization layer for a neural network
    in tensorflow:
    prev: is the activated output of the previous layer
    n: is the number of nodes in the layer to be created
    activation: is the activation function that should be used on the output of
    the layer
    Return: a tensor of the activated output for the layer"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, kernel_initializer=init)(prev)
    mean, variance = tf.nn.moments(layer, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True,
                        name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True,
                       name='beta')
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(layer, mean, variance,
                                           beta, gamma, epsilon)
    return activation(batch_norm)
