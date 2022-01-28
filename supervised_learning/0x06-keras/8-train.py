#!/usr/bin/env python3
"""8. Save Only the Best"""
import tensorflow.keras as K

def train_model(network, data, labels, 
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False, 
                filepath=None, verbose=True, shuffle=False):
    """Updates function 7-train.py"""
    cb = []
    if validation_data is not None:
        if early_stopping:
            early = K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience,
                                              mode='min')
            cb.append(early)
        if learning_rate_decay:
            def decayed_lr(step):
                return alpha / (1 + decay_rate * (step))
            learn = K.callbacks.LearningRateScheduler(schedule=decayed_lr,
                                                      verbose=True)
            cb.append(learn)
        if save_best:
            saver = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
            cb.append(saver)
    history = network.fit(data, labels, epochs=2,
                          batch_size=batch_size,
                          callbacks=cb,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle)
    return history
