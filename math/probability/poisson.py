#!/usr/bin/env python3
"""module poisson"""


class Poisson:
    """class poisson"""

    def __init__(self, data=None, lambtha=1.):
        """constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """calculates the PMF for a given number of successes."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            e = 2.7182818285
            c = 1
            for i in range(1, k + 1):
                c = c * i
            return float((self.lambtha ** k) * (e ** (-self.lambtha)) / c)

    def cdf(self, k):
        """calculates the CDF for a given number of successes."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
