#!/usr/bin/env python3
"15. Slice Like A Ninja"


def np_slice(matrix, axes={}):
    "A function that slices a matrix along specific axes"
    mat_tup_sl = [slice(None, None, None)] * matrix.ndim
    for k, v in sorted(axes.items()):
        sv = slice(*v)
        mat_tup_sl[k] = sv
    matrix = matrix[tuple(mat_tup_sl)]
    return matrix
