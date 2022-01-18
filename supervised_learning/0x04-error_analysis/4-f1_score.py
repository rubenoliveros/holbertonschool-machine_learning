#!/usr/bin/env python3
"""4. F1 score"""
import numpy as np
sensitive = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """A function that calculates the F1 score of a confusion matrix"""
    return 2 / (pow(precision(confusion), -1) + pow(sensitive(confusion), -1))
