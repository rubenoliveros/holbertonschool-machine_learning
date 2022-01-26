#!/usr/bin/env python3
"""1. Input"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """A function that builds a neural network with the Keras library"""
    reg = k.regularizers.l2(lambtha)
    inpt = k.layers.Input(shape=(nx,))
    layer = k.layers.Dense(layers[0],
            input_shape=(nx,),
            activation=activations[0],
            kernel_regularizer=reg)(inpt)

    for i in range(1, len(layers)):
        layer = k.layers.Dropout(rate=1-keep_prob)(layer)
        layer = k.layers.Dense(layers[i],
                activation=activations[i],
                kernel_regularizer=reg)(layer)

    model = k.Model(inputs=inpt, outputs=layer)

    return model
