#!/usr/bin/env python3
"""
17. Integrate
"""


def poly_integral(poly, C=0):
    """
    poly_integral(poly, C=0) - calculates the integral of a polynomial
    @poly: is a list of coefficients representing a polynomial.
           The index of the list represents the power of x that
           the coefficient belongs to.
    @C: is an integer representing the integration constant.
    Returns: a new list of coefficients representing the integral of
             the polynomial.
    """

    if type(poly) is not list or type(C) not in [int, float]:
        return None

    res = [x / (idx + 1) for idx, x in enumerate(poly)]
    res = [
        int(x)
        if type(x) is float and x.is_integer()
        else x
        for x in res
    ]

    if type(C) is float and C.is_integer():
        C = int(C)

    res.insert(0, C)

    while res[-1] == 0:
        del res[-1]

    if len(res) == 0:
        return [0, ]

    return res
