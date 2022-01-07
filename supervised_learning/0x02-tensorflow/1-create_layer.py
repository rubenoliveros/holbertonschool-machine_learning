#!/usr/bin/env python3
"""1. Layers"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """A function for creating a layer"""
    function = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n,
                            name='layer',
                            activation=activation,
                            kernel_initializer=function)
    return layer(prev)
