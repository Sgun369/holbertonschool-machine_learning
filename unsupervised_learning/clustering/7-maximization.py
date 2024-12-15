#!/usr/bin/env python3
"""module maximazation"""
import numpy as np


def maximization(X, g):
    """Performs the maximazation step
    in the EM algorithm
    """
    try:
        if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
            return None, None
        if len(X.shape) != 2 or len(g.shape) != 2:
            return None, None
        n, d = X.shape
        k, n_check = g.shape
        if n != n_check:
            return None, None, None

        if not np.allclose(np.sum(g, axis=0), 1):
            return None, None, None

        N_k = np.sum(g, axis=1)
        pi = N_k / n

        m = np.dot(g, X) / N_k[:, np.newaxis]
        S = np.zeros((k, d, d))

        X_expanded = X[np.newaxis, :, :]
        m_expanded = m[:, np.newaxis, :]

        diff = X_expanded - m_expanded

        S = np.einsum('kn, knd, knd -> kdd', g, diff, diff) / \
            N_k[:, np.newaxis, np.newaxis]
        return pi, m, S
    except Exception as e:
        return None, None, None
