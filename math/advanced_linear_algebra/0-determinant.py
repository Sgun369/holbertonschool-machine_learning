#!/usr/bin/env python3
"""Module to calculate the determinant of a matrix"""


def determinant(matrix):
    """calculate the determinant of a matrix"""
    if not isinstance(
        matrix,
        list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0

    for col in range(len(matrix)):
        minor = [
            [matrix[row][c] for c in range(len(matrix)) if c != col]
            for row in range(1, len(matrix))
        ]
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)

    return det
