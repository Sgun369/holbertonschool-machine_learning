#!/usr/bin/env python3
"""module poisson"""
import math


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
        if k < 0:
            return 0
        k = int(k)
        return (math.exp(-self.lambtha) * (self.lambtha ** k)) / math.factorial(k)
