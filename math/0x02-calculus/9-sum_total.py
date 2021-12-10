#!/usr/bin/env python3
"""Our life is the sum total"""


def summation_i_squared(n):
    """a function that calculates"""
    if type(n) != int or n < 1:
        return None
    result = int((n * (n + 1) * (2 * n + 1)) / 6)
    return result
