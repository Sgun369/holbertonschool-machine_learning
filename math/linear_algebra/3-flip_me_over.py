#!/usr/bin/env python3
"""Flip Me Over"""

def matrix_transpose(matrix):
    """ returns the transpose of a 2D matrix"""
    rows = len(matrix)
    cols = len(matrix[0])

    transpose_matrix = [[None] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transpose_matrix[j][i] = matrix[i][j]
    return transpose_matrix
