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

    def pdf(self, k):
        """Calculates the value of the PDF for a given time period"""
        e = 2.7182818285
        k = int(k)
        if k < 0:
            return 0
        else:
            pdf = self.lambtha * e ** (-self.lambtha * k)
            return pdf

#    def cdf(self, k):
#        """Calculates the value of the CDF for a given number of successes"""
#        k = int(k)
#        if k < 0:
#            return 0
#        cdf = 0
#        for i in range(k + 1):
#            cdf += self.pmf(i)
#        return cdf
