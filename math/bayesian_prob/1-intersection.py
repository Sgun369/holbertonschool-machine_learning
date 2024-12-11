#!/usr/bin/env python3
"""Module intersection"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining x and n
    with the various hypothetical probabilities in P and prior belief in Pr
    """
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
    return intersection
