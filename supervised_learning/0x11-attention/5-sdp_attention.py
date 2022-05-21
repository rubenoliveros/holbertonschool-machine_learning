#!/usr/bin/env python3
"""4. Scaled Dot Product Attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    A function that calculates the scaled dot product attention
    """

    q = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    sca_q = q / tf.math.sqrt(dk)

    if mask is not None:
        sca_q += (mask * -1e9)

    weights = tf.nn.softmax(sca_q, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
