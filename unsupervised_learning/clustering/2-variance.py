#!/usr/bin/env python3
"""module variance"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    try:
        if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
            return None
        if len(X.shape) != 2 or len(C.shape) != 2 or X.shape[1] != C.shape[1]:
            return None
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        min_distances = np.min(distances, axis=1)

        var = np.sum(min_distances ** 2)
        return var
    except Exception as e:
        return None
