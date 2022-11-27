#!/usr/bin/env python3
"""Neural network layer dropout Module"""
# prev is a tensor containing the output of the previous layer
# n is the number of nodes the new layer should contain
# activation is the activation function that should be used on the layer
# keep_prob is the probability that a node will be kept
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that returns output of the layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    Drop_reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=w, name="layer",
                            kernel_regularizer=Drop_reg)
    y = layer(prev)
    return (y)
