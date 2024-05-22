#!/usr/bin/env python3
"""Precision Precision"""
import numpy as np


def precision(confusion):
    """ Calculates the precision for each class in a confusion matrix"""

    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precision = true_positives / (true_positives + false_positives)
    return precision
