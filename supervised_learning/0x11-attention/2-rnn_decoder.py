#!/usr/bin/env python3
"""2. RNN Decoder"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    A class that inherits from tensorflow.keras.layers.Layer
    to decode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Call function
        """
        bat, unds = s_prev.shape
        attention = SelfAttention(unds)
        context, weights = attention(s_prev, hidden_states)
        embeddings = self.embedding(x)
        concat_input = tf.concat([tf.expand_dims(context, 1),
                                  embeddings],
                                 axis=-1)
        outputs, hidden = self.gru(concat_input)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)
        return y, hidden
