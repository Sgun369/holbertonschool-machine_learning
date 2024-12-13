#!/usr/bin/env python3
"""module initialize"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ initializes variables for a Gaussian Mixture Model"""
    try:
        pi = np.full((k,), 1 / k)
        m, _ = kmeans(X, k)
        d = X.shape[1]
        S = np.array([np.eye(d) for _ in range(k)])
        return pi, m, S
    except Exception:
        return None, None, None
