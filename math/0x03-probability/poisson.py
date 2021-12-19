#!/usr/bin/env python3
"""0. Initialize Poisson"""


class Poisson():
    """a class that represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor for class Poison"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if isinstance(data, list):
                if len(data) < 1:
                    raise ValueError("data must contain multiple values")
                else:
                    self.data = data
                    self.lambtha = float(sum(self.data)/len(self.data))
            else:
                raise TypeError("data must be a list")
