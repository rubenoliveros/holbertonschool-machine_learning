#!/usr/bin/env python3
"""3. One Hot"""
import tensorflow.keras as k


def one_hot(labels, classes=None):
    """ A function that converts a label vector into a one-hot matrix"""
    return k.utils.to_categorical(labels, num_classes=classes)
