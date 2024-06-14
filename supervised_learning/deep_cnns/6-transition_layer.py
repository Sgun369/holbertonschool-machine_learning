#!/usr/bin/env python3
"""Transition Layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer"""
    he_normal = K.initializers.HeNormal(seed=0)

    # Batch Normalization
    X = K.layers.BatchNormalization(axis=-1)(X)

    # ReLU activation function
    X = K.layers.ReLU()(X)

    # Calculate the number of filters after compression
    compressed_filters = int(nb_filters * compression)

    # Apply a 1x1 convolution
    X = K.layers.AveragePooling2D(
        pool_size=(
            2, 2), strides=(
            2, 2), padding='same')(X)
    return X, compressed_filters
