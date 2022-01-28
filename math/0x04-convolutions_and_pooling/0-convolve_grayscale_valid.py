#!/usr/bin/env python3
"""0. Valid Convolution"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """A function that performs a valid convolution on grayscale images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    y_h = h - kh + 1
    y_w = w - kw + 1

    c_i = np.zeros((m, y_h, y_w))

    for j in range(y_w):
        for i in range(y_h):
            c_i[:, i, j] = (kernel *
                          images[:, i: i + kh, j: j + kw]).sum(axis=(1, 2))

    return c_i
