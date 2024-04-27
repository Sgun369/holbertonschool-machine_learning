#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if not isinstance(
        poly, list) or any(
        not isinstance(
            coef, (int, float)) for coef in poly):
        return None
    if len(poly) <= 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        derivative.append(poly[power] * power)
    return derivative
