#!/usr/bin/env python3
"""Specificity Specificity"""
import numpy as np


def specificity(confusion):
    """ calculates the specificity
    for each class in a confusion matrix"""
    classes = confusion.shape[0]
    specificities = np.zeros(classes)

    for i in range(classes):
        true_negatives = np.sum(
            np.delete(
                np.delete(
                    confusion,
                    i,
                    axis=0),
                i,
                axis=1))
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]

        specificities[i] = true_negatives / (true_negatives + false_positives)
    return specificities
