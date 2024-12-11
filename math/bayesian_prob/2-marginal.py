#!/usr/bin/env python3
"""Module mrginal"""
import numpy as np


def marginal(x, n, P, Pr):
    """calculates the marginal probability of obtaining x and n"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    likelihood = np.zeros(P.shape)
    for i, p in enumerate(P):
        likelihood = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x)) \
            * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    marginal = np.sum(intersection)
    return marginal
