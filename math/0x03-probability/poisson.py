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
                if len(data) < 2:
                    raise ValueError("data must contain multiple values")
                else:
                    self.data = data
                    self.lambtha = float(sum(self.data)/len(self.data))
            else:
                raise TypeError("data must be a list")

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        e = 2.7182818285
        k = int(k)
        if k < 0:
            return 0
        else:
            k_factorial = 1
            for i in range(1, k + 1):
                k_factorial = k_factorial * i

            pmf = e ** -self.lambtha * self.lambtha ** k / k_factorial
            return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0
        
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf


