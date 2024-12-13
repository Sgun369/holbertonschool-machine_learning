#!/usr/bin/env python3
"""module optimum"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax <= kmin:
        return None, None
    if kmax is None:
        kmax = X.shape[0]

    results = []
    d_vars = []

    C, clss = kmeans(X, kmin, iterations)
    var = variance(X, X)
    if var is None:
        return None, None
    results.append((C, clss))
    d_vars.append(0.0)

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        curr_var = variance(X, C)
        if curr_var is None:
            return None, None
        results.append((C, clss))
        d_vars.append(var - curr_var)
        if var is None:
            continue
        d_vars.append(var)  # Store variance for this k
    return results, d_vars
