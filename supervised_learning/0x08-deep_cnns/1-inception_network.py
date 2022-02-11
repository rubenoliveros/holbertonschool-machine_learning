#!/usr/bin/env python3
"""1. Inception Network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    A function that builds the inception network as
    described in Going Deeper with Convolutions (2014)
    """
    init = K.initializers.he_normal()
    activation = "relu"
    Y = K.Input(shape=(224, 224, 3))
    conv_1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                             strides=(2, 2), padding='same',
                             activation=activation, kernel_initializer=init)(Y)
    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(conv_1)
    conv_2R = K.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                              padding='same', activation=activation,
                              kernel_initializer=init)(max_pool1)
    conv_2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                             padding='same', activation=activation,
                             kernel_initializer=init)(conv_2R)
    max_pool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(conv_2)
    incep_1 = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])
    incep_2 = inception_block(incep_1, [128, 128, 192, 32, 96, 64])
    max_pool3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(incep_2)
    incep_3 = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])
    incep_4 = inception_block(incep_3, [160, 112, 224, 24, 64, 64])
    incep_5 = inception_block(incep_4, [128, 128, 256, 24, 64, 64])
    incep_6 = inception_block(incep_5, [112, 144, 288, 32, 64, 64])
    incep_7 = inception_block(incep_6, [256, 160, 320, 32, 128, 128])
    max_pool4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(incep_7)
    incep_8 = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    incep_9 = inception_block(incep_8, [384, 192, 384, 48, 128, 128])
    avg_pool1 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                          padding='valid')(incep_9)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool1)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=init)(dropout)
    model = K.Model(inputs=Y, outputs=softmax)
    return model
