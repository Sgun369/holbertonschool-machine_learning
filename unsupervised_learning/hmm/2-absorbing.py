#!/usr/bin/env python3
"""Module to determine if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing"""
    if not isinstance(
            P,
            np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    absorbing_states = np.diag(P) == 1

    if not np.any(absorbing_states):
        return False

    if np.all(absorbing_states):
        return True

    transient_indices = np.where(~absorbing_states)[0]
    absorbing_indices = np.where(absorbing_states)[0]

    # Transitions among transient states
    Q = P[np.ix_(transient_indices, transient_indices)]
    # Transitions from transient to absorbing states
    R = P[np.ix_(transient_indices, absorbing_indices)]

    try:
        fundamental_matrix = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    except np.linalg.LinAlgError:
        return False

    reachability = np.matmul(fundamental_matrix, R)

    return np.all(np.sum(reachability, axis=1) > 0)
