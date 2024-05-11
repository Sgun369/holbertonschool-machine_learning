#!/usr/bin/env python3
"""Accuracy"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    predicted_labels = tf.argmax(y_pred, axis=1)

    correct_predictions = tf.equal(predicted_labels, tf.argmax(y, axis=1))

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
