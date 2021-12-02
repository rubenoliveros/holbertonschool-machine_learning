#!/usr/bin/env python3
'''2. Size Me Please'''


def matrix_shape(matrix):
    """Calculates The Matrix's Shape"""
    result = []
    while type(matrix) is list:
        result.append(len(matrix))
        matrix = matrix[0]
    return result
