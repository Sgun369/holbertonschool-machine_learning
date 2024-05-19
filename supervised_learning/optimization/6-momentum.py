#!/usr/bin/env python3
"""Momentum Upgraded"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """sets up the gradient descent with momentum
    optimization algorithm in TensorFlow"""
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
