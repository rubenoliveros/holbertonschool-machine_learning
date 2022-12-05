#!/usr/bin/env python3
"""Module that trains a convolutional neural network to
classify the CIFAR 10 dataset:"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """function thtat pre-processes the data for the model"""
    X = X.astype('float32')
    X_p = K.applications.inception_v3.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    """ Program that trains dataset 32x32 color training images
     and test images"""
    (Xtrain, Ytrain), (X, Y) = K.datasets.cifar10.load_data()
    Xt_p, Yt_p = preprocess_data(Xtrain, Ytrain)
    X_p, Y_p = preprocess_data(X, Y)
    density = K.applications.InceptionV3(include_top=False,
                                         weights='imagenet')
    density.trainable = False
    model = K.Sequential()
    model.add(K.layers.Lambda(lambda x:
                              K.backend.resize_images(x,
                                                      9,
                                                      9,
                                                      'channels_last',
                                                      'bilinear')))
    model.add(density)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(256, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(10, activation=('softmax')))
    callback = []

    def rate_decay(epoch):
        return 0.001 / (1 + (0.01 * epoch))
    learning = K.callbacks.LearningRateScheduler(schedule=rate_decay,
                                                 verbose=1)
    callback.append(learning)
    callback.append(K.callbacks.ModelCheckpoint('cifar10.h5',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max'))
    opt = K.optimizers.SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.fit(x=Xt_p,
              y=Yt_p,
              batch_size=128,
              epochs=30,
              verbose=1,
              shuffle=True,
              validation_data=(X_p, Y_p),
              callbacks=callback)
