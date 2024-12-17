#!/usr/bin/env python3
"""module Markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov"""
    if not isinstance(
            P, np.ndarray) or len(
            P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, P.shape[0]):
        return None
    if not isinstance(t, int) or t < 0:
        return None

    try:
        for _ in range(t):
            s = np.matmul(s, P)
        return s
    except Exception:
        return None
