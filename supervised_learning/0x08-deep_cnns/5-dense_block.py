#!/usr/bin/env python3
"""5. Dense Block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    A function that builds a dense block as described
    in Densely Connected Convolutional Networks
    """
    init = K.initializers.he_normal()
    activ = "relu"

    for i in range(layers):
        normal_1 = K.layers.BatchNormalization()(X)
        activ_1 = K.layers.Activation(activ)(normal_1)
        filt = 4 * growth_rate
        bottleneck = K.layers.Conv2D(filters=filt, kernel_size=(1, 1),
                                     padding='same',
                                     kernel_initializer=init)(activ_1)

        normal_2 = K.layers.BatchNormalization()(bottleneck)
        activ_2 = K.layers.Activation(activ)(normal_2)
        conv = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=init)(activ_2)

        X = K.layers.concatenate([X, conv])
        nb_filters += growth_rate
    return X, nb_filters
