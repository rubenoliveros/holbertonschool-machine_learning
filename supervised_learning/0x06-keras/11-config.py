#!/usr/bin/env python3
"""4-train"""


import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s configuration in JSON format
    network: neural network
    filename: path to file to save the model"""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """Function that load config from json file
    filename: path to json file that contains config of neural network
    return: neural network"""
    with open(filename, 'r') as f:
        config = f.read()
    network = K.models.model_from_json(config)
    return network
