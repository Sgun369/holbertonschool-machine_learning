#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not poly or not isinstance(C, int):
        return None

    integral = [C]

    for i, coef in enumerate(poly):
        if coef != 0:
            new_coef = coef / (i + 1)
            if new_coef.is_integer():
                new_coef = int(new_coef)
            integral.append(new_coef)
        else:
            integral.append(0)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
