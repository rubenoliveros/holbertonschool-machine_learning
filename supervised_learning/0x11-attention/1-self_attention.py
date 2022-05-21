#!/usr/bin/env python3
"""1. Self Attention"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    A class that inherits from tensorflow.keras.layers.Layer
    to calculate the attention for machine translation
    """
    def __init__(self, units):
        """
        Class constructor
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        call function
        """
        s_prev1 = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(s_prev1) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        contx = tf.reduce_sum(weights * hidden_states, axis=1)

        return contx, weights
