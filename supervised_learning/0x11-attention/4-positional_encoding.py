#!/usr/bin/env python3
"""3. Positional Encoding"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    A function that calculates the positional encoding for a transformer
    """
    pos_encoding = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    dm_n = np.float32(dm)

    grad_angle = 1 / (np.power(10000, (2 * (i // 2) / dm_n)))
    angle = pos_encoding * grad_angle

    positional = np.zeros((max_seq_len, dm))

    positional[:, 0::2] = np.sin(angle[:, 0::2])
    positional[:, 1::2] = np.cos(angle[:, 1::2])
    return positional
