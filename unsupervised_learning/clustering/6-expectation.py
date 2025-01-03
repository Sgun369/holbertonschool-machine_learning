#!/usr/bin/env python3
"""Module expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithim"""
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
            return None, None
        if not isinstance(m, np.ndarray) or len(m.shape) != 2:
            return None, None
        if not isinstance(S, np.ndarray) or len(S.shape) != 3:
            return None, None

        n, d = X.shape
        k = pi.shape[0]

        if m.shape != (k, d) or S.shape != (k, d, d) or pi.shape != (k,):
            return None, None

        pdfs = np.array([pdf(X, m[i], S[i]) for i in range(k)])
        weighted_pdfs = pi[:, np.newaxis] * pdfs
        total_pdf = np.sum(weighted_pdfs, axis=0)
        g = weighted_pdfs / total_pdf
        l = np.sum(np.log(total_pdf))
        return g, l
    except Exception as e:
        return None, None
