#!/usr/bin/env python3
"""4. Exponential PDF"""


class Binomial():
    """a class that represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor for Binomial"""
        self.n = int(n)
        self.p = float(p)

        if data is None:
            if self.n < 1:
                raise ValueError("n must be a positive value")
            elif self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.data = data
                    mean = sum(self.data) / len(self.data)
                    variance = (sum([(num - mean) ** 2 for num in self.data]) /
                                len(self.data))
                    p = 1 - variance / mean
                    self.n = int(round(mean / p))
                    self.p = mean / self.n
                else:
                    raise ValueError("data must contain multiple values")

            else:
                raise TypeError("data must be a list")
