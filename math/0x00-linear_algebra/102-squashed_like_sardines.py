#!/usr/bin/env python3
"17. Squashed Like Sardines"


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis
    Args:
        mat1 (list): matrix of int/floats
        mat2 list): matrix of int/floats
        axis (int, optional): axis to perform concatenation. Defaults to 0.
    Returns:
        list: concatenated matrix. None if it can't be done
    """
    # check if columns are equals
    if not axis and matrix_shape(mat1) == matrix_shape(mat2):
        return mat1 + mat2
    # check if rows are equals
    if axis == 1 and len(mat1) == len(mat2):
        return [mat1[row] + mat2[row] for row in range(len(mat1))]

    if axis == 3 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2


def matrix_shape(matrix):
    """Calculates the shape of a matrix with the same dimension
    Args:
        matrix (list): matrix to be evaluated
    Returns:
        shape_matrix (list): list of integers desribing the matrix shape
    """
    # base case
    if type(matrix) is int:
        return []

    shape_matrix = matrix_shape(matrix[0])
    shape_matrix.insert(0, len(matrix))
    return shape_matrix
