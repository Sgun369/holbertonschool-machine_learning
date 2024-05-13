#!/usr/bin/env python3
"""module Train_Op"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    training = optimizer.minimize(loss)
    return training
