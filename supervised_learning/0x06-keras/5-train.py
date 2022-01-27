#!/usr/bin/env python3
"""5. Validate"""
import tensorflow.keras as k


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                verbose=True,
                shuffle=False):
    """Updates the function on 4-train.py to also analyze validation data"""
    return network.fit(data,
                       labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=verbose,
                       shuffle=shuffle)
