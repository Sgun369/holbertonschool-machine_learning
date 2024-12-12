#!/usr/bin/env python3
"""module k-means"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """perform K-means clustering on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    clss = np.zeros(n, dtype=int)

    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_new = np.zeros_like(C)
        for j in range(k):
            points_in_cluster = X[clss == j]

            if points_in_cluster.shape[0] == 0:
                C_new[j] = np.random.uniform(
                    low=min_vals, high=max_vals, size=(d,))
            else:
                C_new[j] = np.mean(points_in_cluster, axis=0)

        if np.all(C == C_new):
            break
        C = C_new
    return C, clss
