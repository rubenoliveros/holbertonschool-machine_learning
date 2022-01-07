#!/usr/bin/env python3
"""4. Loss"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """A function that calculates the softmax cross-entropy loss"""
    loss = tf.losses.softmax_cross_entropy(
        y, y_pred, reduction=tf.losses.Reduction.MEAN)
    return loss
