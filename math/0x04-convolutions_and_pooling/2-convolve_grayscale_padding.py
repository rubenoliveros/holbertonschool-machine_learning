#!/usr/bin/env python3
"""2. Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    A function that performs a same convolution on grayscale 
    images with custom padding
    """
    x_w, x_h, m = images.shape[2], images.shape[1], images.shape[0]
    kernel_w, kernel_h = kernel.shape[1], kernel.shape[0]
    pad_w, pad_h = padding[1], padding[0]
    y_w = x_w + 2 * pad_w - kernel_w + 1
    y_h = x_h + 2 * pad_h - kernel_h + 1

    image = np.pad(images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                   mode='constant')

    y = np.zeros((m, y_h, y_w))

    for j in range(y_w):
        for i in range(y_h):
            y[:, i, j] = (kernel *
                          image[:, i: i + kernel_h,
                                j: j + kernel_w]).sum(axis=(1, 2))

    return y
