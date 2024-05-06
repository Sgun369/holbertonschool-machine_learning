#!/usr/bin/env python3
"""Placeholders"""
import tensorflow.compact.v1 as tf  # type: ignore


def create_placeholders(nx, classes):
    """returns two placeholders, x and y,
    for the neural network"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
