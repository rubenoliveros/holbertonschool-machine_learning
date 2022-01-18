#!/usr/bin/env python3
"""3. Specificity"""
import numpy as np


def specificity(confusion):
    """A function that calculates the specificity in a confusion matrix"""
    ALL = np.sum(confusion)
    TP = np.diag(confusion)
    PP = np.sum(confusion, axis=0)
    P = np.sum(confusion, axis=1)
    return (ALL - PP - P + TP) / (ALL - P)
