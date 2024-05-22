#!/usr/bin/env python3
"""Sensitivity Sensitivity"""

import numpy as np


def sensitivity(confusion):
    """ Calculates the sensitivity for each class in a confusion matrix"""

    true_positives = np.diag(confusion)

    false_negatives = np.sum(confusion, axis=1) - true_positives

    sensitivity = true_positives / (true_positives + false_negatives)

    return sensitivity
