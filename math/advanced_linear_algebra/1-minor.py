#!/usr/bin/env python3
"""
 module to compute the minor matrix of a given square matrix.
"""


def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): A square matrix represented as a list of lists.

    Returns:
        list of lists: The minor matrix of the input matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
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

    if rows == 1:
        return [[1]]

    def determinant(sub_matrix):
        """Calculates the determinant of a square matrix recursively."""
        if len(sub_matrix) == 1:
            return sub_matrix[0][0]
        if len(sub_matrix) == 2:
            return sub_matrix[0][0] * sub_matrix[1][1] - \
                sub_matrix[0][1] * sub_matrix[1][0]

        det = 0
        for col in range(len(sub_matrix)):
            minor = [row[:col] + row[col + 1:] for row in sub_matrix[1:]]
            det += ((-1) ** col) * sub_matrix[0][col] * determinant(minor)
        return det

    minor_matrix = []
    for i in range(rows):
        minor_row = []
        for j in range(rows):
            sub_matrix = [row[:j] + row[j + 1:]
                          for idx, row in enumerate(matrix) if idx != i]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
