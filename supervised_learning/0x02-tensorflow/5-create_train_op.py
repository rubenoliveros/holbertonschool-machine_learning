#!/usr/bin/env python3
"""5. Train_Op"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """A function that creates the training operation for the network"""
    operation = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return operation
