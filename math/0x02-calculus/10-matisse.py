#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    """a function that calculates the derivative of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if poly is None:
        return None
    if len(poly) == 1:
        return [0]
    return [i * c for i, c in enumerate(poly) if i]
