#!/usr/bin/env python3
"""6. Transition Layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    A function that builds a transition layer as described
    in Densely Connected Convolutional Networks
    """
    init = K.initializers.he_normal()
    activ = "relu"
    filt = int(nb_filters * compression)

    normal_1 = K.layers.BatchNormalization()(X)
    activ_1 = K.layers.Activation(activ)(normal_1)
    conv_1 = K.layers.Conv2D(filters=filt, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=init)(activ_1)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         padding='same')(conv_1)
    return avg_pool, filt
