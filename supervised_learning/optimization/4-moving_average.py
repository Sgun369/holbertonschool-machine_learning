#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    moving_averages = []
    v = 0
    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        bias_correction = 1 - beta ** t
        corrected_v = v / bias_correction
        moving_averages.append(corrected_v)
    return moving_averages
