#!/usr/bin/env python3
"""Layers"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Creates a neural network layer with
    He initialization"""

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer')

    return layer(prev)
