#!/usr/bin/env python3
"""module that calculates the definnitness of a matrix"""

import numpy as np


def definiteness(matrix):
    """calculate the definitness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None
    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    return None
