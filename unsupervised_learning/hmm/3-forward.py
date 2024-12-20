#!/usr/bin/env python3
"""Module forward"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hdden Markow model"""
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        F = np.zeros((N, T))

        F[:, 0] = Initial.ravel() * Emission[:, Observation[0]]

        for t in range(1, T):
            for j in range(N):
                F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]
                                 ) * Emission[j, Observation[t]]

                P = np.sum(F[:, -1])
                return P, F
    except Exception as e:
        return None, None
