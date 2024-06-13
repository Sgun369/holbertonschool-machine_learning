#!/usr/bin/env python3
"""Projection Block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """ builds a projection block
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.HeNormal(seed=0)

    # First 1x1 convolution in the main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(
        s, s), padding='same', kernel_initializer=he_normal)(A_prev)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)

    # 3x3 convolution in the main path
    X = K.layers.Conv2D(
        filters=F3, kernel_size=(
            3, 3), strides=(
            1, 1), padding='same', kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)

    # Second 1x1 convolution in the main path
    X = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            1, 1), padding='same', kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)

    # 1x1 convolution in the shortcut path
    shortcut = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            s, s), padding='same', kernel_initializer=he_normal)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=-1)(shortcut)

    # Add shortcut value to the main path, and pass it through a ReLU
    # activation
    X = K.layers.Add()([X, shortcut])
    X = K.layers.ReLU()(X)

    return X
