#!/usr/bin/env python3
"""Bracing The Elements"""


def np_elementwise(mat1, mat2):
    """ a function def np_elementwise(mat1, mat2): that performs element-wise
    addition,
    subtraction, multiplication, and division"""
    sum = mat1 + mat2
    diff = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return sum, diff, mul, div
