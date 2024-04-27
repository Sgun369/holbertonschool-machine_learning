#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    integral = [C]

    for i in range(len(poly)):
        if poly[i] != 0:
            new_coeff = poly[i] / (i + 1)
            if new_coeff.is_integer():
                new_coeff = int(new_coeff)
            integral.append(new_coeff)
        else:
            integral.append(0)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
