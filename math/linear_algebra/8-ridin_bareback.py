#!/usr/bin/env python3
"""Ridin’ Bareback"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    # check if matrices can be mltiplied
    if len(mat1[0]) != len(mat2):
        return None

    # initialize result matrix with appropriate dimensions
    result = [[0] * len(mat2[0]) for _ in range(len(mat1))]

    # perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
