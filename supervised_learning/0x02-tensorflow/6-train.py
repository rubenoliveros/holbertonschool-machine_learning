#!/usr/bin/env python3
"""6. Train"""
import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
        iterations, save_path="/tmp/model.ckpt"):
    """A function that builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    session = tf.global_variables_initializer()
    store = tf.train.Saver()

    with tf.Session() as trainer:
        trainer.run(session)
        for i in range(iterations + 1):
            train_cost = trainer.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = trainer.run(
                accuracy, feed_dict={
                    x: X_train, y: Y_train})
            valid_cost = trainer.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_acc = trainer.run(
                accuracy, feed_dict={
                    x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))

            if i < iterations:
                trainer.run(train_op, feed_dict={x: X_train, y: Y_train})
        return store.save(trainer, save_path)
