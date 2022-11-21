#!/usr/bin/env python3
"16. The Whole Barn"


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


def matrix_add_recursive(matrix1, matrix2):
    """Add a recursive two matrices"""
    retorno = []
    for i in range(len(matrix1)):
        if type(matrix1[i]) == list:
            retorno.append(matrix_add_recursive(matrix1[i], matrix2[i]))
        else:
            retorno.append(matrix1[i] + matrix2[i])
    return retorno


def add_matrices(mat1, mat2):
    """Adds two matrices and checks if they are equal"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        return matrix_add_recursive(mat1, mat2)
