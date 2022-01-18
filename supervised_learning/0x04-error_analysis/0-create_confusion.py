#!/usr/bin/env python3
"""0. Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """A function that creates a confusion matrix"""
    return np.matmul(labels.T, logits)
