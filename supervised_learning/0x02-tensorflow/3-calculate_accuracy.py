#!/usr/bin/env python3
"""3. Accuracy"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """A function that calculates the accuracy of a prediction"""
    validate = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    acc = tf.reduce_mean(tf.cast(validate, tf.float32))
    return acc
