#!/usr/bin/env python3
"""module of the class Normal"""


class Normal:
    """class Normal"""

    def __init__(self, data=None, mean=0, stddev=1.):
        """constructor of the class Normal"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calculates the c-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """calculates the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        return coefficient * e ** exponent
