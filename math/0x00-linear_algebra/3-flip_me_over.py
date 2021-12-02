#!/usr/bin/env python3
"""Flip Me Over"""


def matrix_transpose(matrix):
    """a function that returns the transpose of a 2D matrix"""
    zipped_rows = zip(*matrix)
    transpose_matrix = [list(row) for row in zipped_rows]
    return transpose_matrix
