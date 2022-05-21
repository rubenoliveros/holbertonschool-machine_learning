#!/usr/bin/env python3
"""0. RNN Encoder"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    A class that inherits from tensorflow.keras.layers.Layer
    to encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """Class constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        """
        initializer = tf.keras.initializers.Zeros()
        hiddenQ = initializer(shape=(self.batch, self.units))
        return hiddenQ

    def call(self, x, initial):
        """
        Call methods
        """
        embedding = self.embedding(x)
        outputs, last_hiddenQ = self.gru(embedding,
                                         initial_state=initial)
        return outputs, last_hiddenQ
