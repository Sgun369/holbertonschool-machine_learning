#!/usr/bin/env python3
"""module of the class Binomial"""


class Binomial:
    """class Binomial"""

    def __init__(self, data=None, n=1, p=0.5):
        """constructor of the class Binomial"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        """calculates the value of the PMF for a given number of successes"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        binomial_coef = 1
        for i in range(1, k + 1):
            binomial_coef *= (self.n - (k - i)) / i

        pmf_value = binomial_coef * \
            (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf_value
