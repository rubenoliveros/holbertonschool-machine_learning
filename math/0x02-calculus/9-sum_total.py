#!/usr/bin/env python3
"""Our life is the sum total"""


def summation_i_squared(n):
    """a function that calculates \sum_{i=1}^{n} i^2"""
    if type(n) != int or n < 1:
        return None
    result = int((n * (n + 1) * (2 * n + 1)) / 6)
    return result
