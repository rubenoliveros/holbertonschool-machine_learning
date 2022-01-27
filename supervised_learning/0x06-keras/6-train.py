#!/usr/bin/env python3
"6. Early Stopping"
import tensorflow.keras as k


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                verbose=True,
                shuffle=False):
    """Updates the function 5-train.py to also train the model using early stopping"""
    es = None
    if validation_data is not None and early_stopping:
        es = [k.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
    return network.fit(data,
                       labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       callbacks=es,
                       verbose=verbose,
                       shuffle=shuffle)
