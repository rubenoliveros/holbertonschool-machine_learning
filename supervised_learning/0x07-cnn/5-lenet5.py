#!/usr/bin/env python3
"""5. LeNet-5 (Keras)"""
import tensorflow.keras as K


def lenet5(X):
    """
    A function that builds a modified version of
    the LeNet-5 architecture using keras
    """
    init = K.initializers.he_normal(seed=None)

    conv_1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                             activation='relu', kernel_initializer=init)(X)
    pool_1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_1)

    conv_2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             activation='relu',
                             kernel_initializer=init)(pool_1)
    pool_2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_2)

    flat = K.layers.Flatten()(pool_2)

    full_1 = K.layers.Dense(120, activation='relu',
                            kernel_initializer=init)(flat)
    full_2 = K.layers.Dense(84, activation='relu',
                            kernel_initializer=init)(full_1)

    soft_out = K.layers.Dense(10, activation='softmax',
                              kernel_initializer=init)(full_2)

    model = K.models.Model(X, soft_out)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
