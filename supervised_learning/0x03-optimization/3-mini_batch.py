#!/usr/bin/env python3
"""Shuffle data"""


import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Function that trains a loaded neural network model using mini-batch
    gradient descent:
    X_train: is a numpy.ndarray of shape (m, 784) containing the training data
        m is the number of data points
        784 is the number of input features
    Y_train: is a one-hot numpy.ndarray of shape (m, 10) containing the
    training labels
        10 is the number of classes the model should classify
    X_valid: is a numpy.ndarray of shape (m, 784) containing the validation
    data
    Y_valid: is a one-hot numpy.ndarray of shape (m, 10) containing the
    validation labels
    batch_size: is the number of data points in a batch
    epochs: is the number of times the training should pass through the whole
    dataset
    load_path: is the path from which to load the model
    save_path: is the path to where the model should be saved after training
    Returns: the path where the model was saved"""
    with tf.Session() as sess:
        metaGraph = tf.train.import_meta_graph(load_path + ".meta")
        metaGraph.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection("y")[0]
        loss = tf.get_collection('loss')[0]
        acc = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]
        feed_train = {x: X_train, y: Y_train}
        feed_valid = {x: X_valid, y: Y_valid}

        batch_float = X_train.shape[0] / batch_size
        batch_int = int(batch_float)

        extra = False
        # conditional that identify if is necessary an extra step
        if (batch_float > batch_int):
            batch_int = int(batch_float) + 1
            extra = True

        for epoch in range(epochs + 1):
            cost_train = sess.run(loss, feed_train)
            acc_train = sess.run(acc, feed_train)
            cost_valid = sess.run(loss, feed_valid)
            acc_valid = sess.run(acc, feed_valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(acc_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(acc_valid))

            if (epoch < epochs):
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for i in range(batch_int):
                    start = i * batch_size
                    end = batch_size * (i + 1)
                    if i == batch_int - 1 and extra:
                        end = int(batch_size * (
                            i + batch_float - batch_int + 1))
                    feed_batch = {x: X_shuffled[start: end],
                                  y: Y_shuffled[start: end]}
                    sess.run(train_op, feed_batch)
                    if i != 0 and (i + 1) % 100 == 0:
                        print("\tStep {}:".format(i + 1))
                        cost_batch = sess.run(loss, feed_batch)
                        print("\t\tCost: {}".format(cost_batch))
                        acc_batch = sess.run(acc, feed_batch)
                        print("\t\tAccuracy: {}".format(acc_batch))
        save_path = metaGraph.save(sess, save_path)
    return save_path
