#!/usr/bin/env python3
"""Cat's Got Your Tongue"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ a function def np_cat(mat1, mat2, axis=0)
    that concatenates two matrices
    along a specific axis:"""
    return np.concatenate((mat1, mat2), axis=axis)
