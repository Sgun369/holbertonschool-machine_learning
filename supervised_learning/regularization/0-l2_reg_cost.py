#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network
    with L2 regularization"""
    l2_term = 0
    for i in range(1, L + 1):
        weight_matrix = weights[f'W{i}']
        l2_term += np.sum(np.square(weight_matrix))

    l2_term = (lambtha / (2 * m)) * l2_term
    l2_reg_cost = cost + l2_term
    return l2_reg_cost
