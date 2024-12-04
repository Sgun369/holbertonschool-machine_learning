#!/usr/bin/env python3
"""module to calculate the inverse of a matrix"""

adjugate = __import__('3-adjugate').adjugate
determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """calculates the inverse of a matrix"""
    if not isinstance(
        matrix,
        list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty suare matrix")

    det = determinant(matrix)
    if det == 0:
        return None
    adj = adjugate(matrix)

    n = len(matrix)
    inverse_matrix = [[adj[i][j] / det for j in range(n)] for i in range(n)]
    return inverse_matrix
