#!/usr/bin/env python3
"""1. Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """A function that performs a same convolution on grayscale images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    y_h = h - kh + 1
    y_w = w - kw + 1

    if kh % 2 == 0:
        pad_h = int(kh / 2)
    else:
        pad_h = int((kh - 1) / 2)
    if kw % 2 == 0:
        pad_w = int(kw / 2)
    else:
        pad_w = int((kw - 1) / 2)

    image = np.pad(images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                   mode='constant')

    c_i = np.zeros((m, y_h, y_w))

    for j in range(y_w):
        for i in range(y_h):
            c_i[:, i, j] = (kernel *
                            images[:, i: i + kh, j: j + kw]).sum(axis=(1, 2))

    return c_i
