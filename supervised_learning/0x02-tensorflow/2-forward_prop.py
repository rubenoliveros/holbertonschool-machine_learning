#!/usr/bin/env python3
"""2. Forward Propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """A function that creates the forward propagation graph for the NN"""
    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
