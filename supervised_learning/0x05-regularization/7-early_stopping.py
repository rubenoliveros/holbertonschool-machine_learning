#!/usr/bin/env python3
"""7. Early Stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """A function that determines if you should stop gradient descent early"""
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        boolean = True
    else:
        boolean = False
    return boolean, count
