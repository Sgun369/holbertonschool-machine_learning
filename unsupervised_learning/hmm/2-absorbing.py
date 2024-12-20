#!/usr/bin/env python3
"""Module to determine if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n, m = P.shape
    if n != m:
        return False

    absorbing_states = np.diag(P) == 1

    if not np.any(absorbing_states):
        return False

    if np.all(absorbing_states):
        return True

    transient_states = ~absorbing_states

    indices = np.argsort(absorbing_states)[::-1]
    P_canonical = P[indices][:, indices]

    num_absorbing = np.sum(absorbing_states)
    R = P_canonical[num_absorbing:, :num_absorbing]
    Q = P_canonical[num_absorbing:, num_absorbing:]

    identity = np.eye(Q.shape[0])
    try:
        fundamental_matrix = np.linalg.inv(identity - Q)
    except np.linalg.LinAlgError:
        return False

    reachability_matrix = np.matmul(fundamental_matrix, R)
    return np.all(reachability_matrix > 0)
