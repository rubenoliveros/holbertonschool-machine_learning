#!/usr/bin/env python3
"17. Squashed Like Sardines"


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.
    Parameters
    ----------
    matrix : array-like
        The matrix to calculate the shape of
    Returns
    -------
    int
        The shape of the of matrix
    """
    if type(matrix[0]) != list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])


def matrix_creation(mat1, mat2, axis=0):
    "If possible, creates a new matrix for concatenation"
    retorno = []
    if axis == 0:
        retorno = mat1 + mat2
        return retorno
    for x in range(len(mat1)):
        retorno.append(matrix_creation(mat1[x], mat2[x], axis - 1))
    return retorno


def cat_matrices(mat1, mat2, axis=0):
    "Concatenates two matrices"
    uno = len(matrix_shape(mat1))
    dos = len(matrix_shape(mat2))
    if uno >= axis and dos >= axis and uno == dos:
        return matrix_creation(mat1, mat2, axis)
    else:
        return None
