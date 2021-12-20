#!/usr/bin/env python3
"""4. Exponential PDF"""

e = 2.7182818285

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

    def pdf(self, k):
        """Calculates the value of the PDF for a given time period"""
        k = int(k)
        if k < 0:
            return 0
        else:
            return self.lambtha * (e ** (-self.lambtha * k))

    def cdf(self, k):
        """ Calculates the value of the CDF for a given time period"""
        k = int(k)
        if k < 0:
            return 0
        cdf = -e ** (-self.lambtha * k) + 1
        return cdf
