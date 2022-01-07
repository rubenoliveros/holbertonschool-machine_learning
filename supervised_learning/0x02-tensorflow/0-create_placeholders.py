#!/usr/bin/env python3
"""0. Placeholders"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """A function that returns two placeholders for the neural network"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
