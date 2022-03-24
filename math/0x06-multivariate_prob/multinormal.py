#!/usr/bin/env python3
"""2. Initialize - 3. PDF"""


import numpy as np


class MultiNormal:
    """A class that represents a Multivariate Normal distribution"""
    def __init__(self, data):
        """
        Class constructor
        parameters:
            data [numpy.ndarray of shape (d, n)]:
                contains the data set
                d: number of dimensions in each data point
                n: number of data points
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov
