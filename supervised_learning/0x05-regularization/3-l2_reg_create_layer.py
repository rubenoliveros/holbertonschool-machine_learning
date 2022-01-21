#!/usr/bin/env python3
"""3. Create a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """A function that creates a tensorflow layer that includes L2 
    regularization"""
    function = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regular = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(
        units=n,
        name='layer',
        activation=activation,
        kernel_initializer=function,
        kernel_regularizer=regular)
    return layer(prev)
