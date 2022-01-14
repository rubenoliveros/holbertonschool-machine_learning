#!/usr/bin/env python3
"""1. Normalize"""


def normalize(X, m, s):
    """A function that normalizes a matrix"""
    norm = (X - m) / s
    return norm
