#!/usr/bin/env python3
"""6. Initialize Normal"""


class Normal():
    """a class Normal that represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor for class Normal"""
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.mean = sum(data) / len(data)
                    n = len(data)
                    variance = sum([(n - self.mean) ** 2 for n in data]) / n
                    self.stddev = variance ** 0.5
                else:
                    raise ValueError("data must contain multiple values")

            else:
                raise TypeError("data must be a list")
