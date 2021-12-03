#!/usr/bin/env python3
"""Ridin’ Bareback"""


def mat_mul(mat1, mat2):
    """a function that performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    new = []
    for i in range(len(mat1)):
        new.append([])
        for j in range(len(mat2[0])):
            result = 0
            for k in range(len(mat1[0])):
                result += mat1[i][k] * mat2[k][j]
            new[i].append(result)
    return new
