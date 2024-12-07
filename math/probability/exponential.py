#!/usr/bin/env python3
"""module of the class Exponential"""


class Exponential:
    """class Exponential"""

    def __init__(self, data=None, lambtha=1.):
        """constructor of the class Exponential"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean_data = sum(data) / len(data)
            self.lambtha = 1 / mean_data

    def pdf(self, x):
        """calculates the PDF for a given value of x"""
        if x < 0:
            return 0
        else:
            e = 2.7182818285
            return self.lambtha * e ** (-self.lambtha * x)
    
    
    def cdf(self, x):
        """Calculates theXDF for agiven time periode x"""
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - e ** (-self.lambtha * x)
