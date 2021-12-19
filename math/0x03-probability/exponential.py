#!/usr/bin/env python3
"""3. Initialize Exponential"""


class Exponential():
    """a class that represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor for class Exponential"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if isinstance(data, list):
                if len(data) < 2:
                    raise ValueError("data must contain multiple values")
                else:
                    self.data = data
                    self.lambtha = float(1 / (sum(self.data) / len(self.data)))
            else:
                raise TypeError("data must be a list")
