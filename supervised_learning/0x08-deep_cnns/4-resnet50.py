#!/usr/bin/env python3
"""4. ResNet-50"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    A function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015)
    """
    init = K.initializers.he_normal()
    activ = "relu"
    Y = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                             strides=(2, 2), padding='same',
                             kernel_initializer=init)(Y)
    normal_1 = K.layers.BatchNormalization()(conv_1)
    activ_1 = K.layers.Activation(activ)(normal_1)
    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(activ_1)

    conv_2x1 = projection_block(max_pool1, [64, 64, 256], 1)
    conv_2x2 = identity_block(conv_2x1, [64, 64, 256])
    conv_2x3 = identity_block(conv_2x2, [64, 64, 256])

    conv_3x1 = projection_block(conv_2x3, [128, 128, 512])
    conv_3x2 = identity_block(conv_3x1, [128, 128, 512])
    conv_3x3 = identity_block(conv_3x2, [128, 128, 512])
    conv_3x4 = identity_block(conv_3x3, [128, 128, 512])

    conv_4x1 = projection_block(conv_3x4, [256, 256, 1024])
    conv_4x2 = identity_block(conv_4x1, [256, 256, 1024])
    conv_4x3 = identity_block(conv_4x2, [256, 256, 1024])
    conv_4x4 = identity_block(conv_4x3, [256, 256, 1024])
    conv_4x5 = identity_block(conv_4x4, [256, 256, 1024])
    conv_4x6 = identity_block(conv_4x5, [256, 256, 1024])

    conv_5x1 = projection_block(conv_4x6, [512, 512, 2048])
    conv_5x2 = identity_block(conv_5x1, [512, 512, 2048])
    conv_5x3 = identity_block(conv_5x2, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding='valid')(conv_5x3)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=init)(avg_pool)
    model = K.Model(inputs=Y, outputs=softmax)
    return model
