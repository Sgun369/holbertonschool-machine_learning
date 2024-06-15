#!/usr/bin/env python3
"""Transition Layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in DenseNet"""
    he_normal = K.initializers.HeNormal(seed=0)

    # Calculate the number of filters after compression
    nb_filters = int(nb_filters * compression)

    # Batch Normalization followed by ReLU activation
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)

    # 1x1 Convolution with compression factor
    X = K.layers.Conv2D(nb_filters, (1, 1), padding='same',
                        kernel_initializer=he_normal)(X)

    # 2x2 Average Pooling
    X = K.layers.AveragePooling2D(
        pool_size=(
            2, 2), strides=(
            2, 2), padding='same')(X)

    return X, nb_filters
