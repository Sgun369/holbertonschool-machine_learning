#!/usr/bin/env python3
"""F1 score"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix"""
    sens = sensitivity(confusion)
    prec = precision(confusion)

    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)

    for i in range(classes):
        if sens[i] + prec[i] == 0:
            f1_scores[i] = 0
        else:
            f1_scores[i] = 2 * (prec[i] * sens[i]) / (prec[i] + sens[i])
    return f1_scores
