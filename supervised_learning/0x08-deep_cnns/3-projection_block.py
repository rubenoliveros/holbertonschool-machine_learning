#!/usr/bin/env python3
"""3. Projection Block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    A function that builds a projection block as
    described in Deep Residual Learning for Image Recognition (2015)
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    activ = "relu"

    conv_11 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=s,
                              padding='same', kernel_initializer=init)(A_prev)
    normal_11 = K.layers.BatchNormalization()(conv_11)
    activ_11 = K.layers.Activation(activ)(normal_11)

    conv_3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                             kernel_initializer=init)(activ_11)
    normal_3 = K.layers.BatchNormalization()(conv_3)
    activ_3 = K.layers.Activation(activ)(normal_3)

    conv_12 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                              padding='same', kernel_initializer=init)(activ_3)
    normal_12 = K.layers.BatchNormalization()(conv_12)

    conv_4f = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=s,
                              padding='same', kernel_initializer=init)(A_prev)
    normal_4f = K.layers.BatchNormalization()(conv_4f)

    added = K.layers.Add()([normal_12, normal_4f])
    activat = K.layers.Activation(activ)(added)
    return activat
