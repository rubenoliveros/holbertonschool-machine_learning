#!/usr/bin/env python3
"""Across The Planes"""

def add_matrices2D(mat1, mat2):
    """a function that adds two matrices element-wise"""
    add = []
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        add.append(list())
        for j in range(len(mat1[i])):
            add[i].append(mat1[i][j] + mat2[i][j])
    return add
