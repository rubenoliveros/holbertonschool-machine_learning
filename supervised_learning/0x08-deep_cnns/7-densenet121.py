#!/usr/bin/env python3
"""7. DenseNet-121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    A function that builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks
    """
    init = K.initializers.he_normal()
    activ = "relu"
    Y = K.Input(shape=(224, 224, 3))
    filt = 2 * growth_rate

    normal_1 = K.layers.BatchNormalization()(Y)
    activ_1 = K.layers.Activation(activ)(normal_1)
    conv_1 = K.layers.Conv2D(filters=filt, kernel_size=(7, 7), strides=(2, 2),
                             padding='same', kernel_initializer=init)(activ_1)
    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(conv_1)

    dense_1, filt = dense_block(max_pool1, filt, growth_rate, 6)
    trans_1, filt = transition_layer(dense_1, filt, compression)

    dense_2, filt = dense_block(trans_1, filt, growth_rate, 12)
    trans_2, filt = transition_layer(dense_2, filt, compression)

    dense_3, filt = dense_block(trans_2, filt, growth_rate, 24)
    trans_3, filt = transition_layer(dense_3, filt, compression)

    dense_4, filt = dense_block(trans_3, filt, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding='valid')(dense_4)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=init)(avg_pool)
    model = K.Model(inputs=Y, outputs=softmax)
    return model
