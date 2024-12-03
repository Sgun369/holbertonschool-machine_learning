#!/usr/bin/env python3
"""module to calculate the cofactor of a matrix"""

determinant = __import__('0-determinant').determinant


def cofactor(matrix):
    """
    Calculates the cofacttor matrix of  a given square matrix
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

    cofactor_matrix = []
    for i in range(rows):
        cofactor_row = []
        for j in range(rows):
            # Create the minor matrix excluding the i-th row and j-th column
            sub_matrix = [row[:j] + row[j + 1:]
                          for idx, row in enumerate(matrix) if idx != i]
            # Calculate cofactor value with the appropriate sign
            cofactor_value = ((-1) ** (i + j)) * determinant(sub_matrix)
            cofactor_row.append(cofactor_value)
        # Append the completed row to the matrix
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix
