#!/usr/bin/env python3
"""2. Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    A function that performs a same convolution on grayscale 
    images with custom padding
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    pad_h, pad_w = padding[0], padding[1]
    y_h = h + 2 * pad_h - kh + 1
    y_w = w + 2 * pad_w - kw + 1

    image = np.pad(images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                   mode='constant')

    c_i = np.zeros((m, y_h, y_w))

    for j in range(y_w):
        for i in range(y_h):
            c_i[:, i, j] = (kernel * images[:, i: i + kh, j: j + kw]).sum(axis=(1, 2))

    return c_i
