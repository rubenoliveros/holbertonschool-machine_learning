#!/usr/bin/env python3
"""5. Bidirectional Cell Forward"""


import numpy as np


class BidirectionalCell():
    """A class that represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """A Bidirectional Cell class constructor"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Method that calculates the hidden state in the forward direction"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(concat, self.Whf) + self.bhf
        h_next = np.tanh(h_next)

        return h_next
