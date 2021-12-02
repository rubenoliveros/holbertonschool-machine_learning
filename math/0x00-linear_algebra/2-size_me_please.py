#!/usr/bin/env python3
'''Size Me Please'''


def matrix_shape(matrix):
    """a function that calculates the shape of a matrix"""
    result = []
    while type(matrix) is list:
        result.append(len(matrix))
        matrix = matrix[0]
    return result
