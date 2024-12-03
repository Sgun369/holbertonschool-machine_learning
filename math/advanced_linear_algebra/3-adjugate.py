#!/usr/bin/env python3
"""module to calculate the adjudent matrix of a matrix"""

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """calculates the adjugate matrix of a given square matrix"""
    if not isinstance(
        matrix,
        list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    rows = len(matrix)
    if rows == 0 or any(len(row) != rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = cofactor(matrix)

    adjugate_matrix = [[cofactor_matrix[i][j]
                        for j in range(rows)] for i in range(rows)]

    return adjugate_matrix
