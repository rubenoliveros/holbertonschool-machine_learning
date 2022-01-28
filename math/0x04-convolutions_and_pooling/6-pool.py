#!/usr/bin/env python3
"""6. Poolin"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """A function  that performs pooling on images"""

    x_w, x_h, m = images.shape[2], images.shape[1], images.shape[0]
    c = images.shape[3]
    kernel_w, kernel_h = kernel_shape[1], kernel_shape[0]
    stride_w, stride_h = stride[1], stride[0]

    y_w = int((x_w - kernel_w) / stride_w + 1)
    y_h = int((x_h - kernel_h) / stride_h + 1)

    y = np.zeros((m, y_h, y_w, c))

    if mode == 'avg':
        pooling = np.average
    else:
        pooling = np.max

    for j in range(y_w):
        for i in range(y_h):
            y[:, i, j, :] =\
                pooling(images[:,
                               i * stride_h: i * stride_h + kernel_h,
                               j * stride_w: j * stride_w + kernel_w,
                               :], axis=(1, 2))

    return y
