#!/usr/bin/env python3
"""Placeholders"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """returns two placeholders, x and y,
    for the neural network"""
    x = tf.placeholder(dtype="float", shape=(None, nx), name='x')
    y = tf.placeholder(dtype="float", shape=(None, classes), name='y')
    return x, y
