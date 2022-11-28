#!/usr/bin/env python3
"""4-train"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """function that makes a prediction using a neural network
    network: is the network model to make the prediction with
    data: is the input data to make the prediction with
    verbose: is a boolean that determines if output should be printed during
    the prediction process
    Returns: the prediction for the data"""
    result = network.predict(
        x=data,
        batch_size=None,
        verbose=verbose
    )
    return result
