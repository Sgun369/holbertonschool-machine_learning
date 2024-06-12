#!/usr/bin/env python3
"""Inception Block"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """ builds an inception block as described in
    Going Deeper with Convolutions (2014)"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1x1 = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu')(A_prev)
    conv3x3_reduce = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu')(conv3x3_reduce)
    conv5x5_reduce = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu')(conv5x5_reduce)
    maxpool = K.layers.MaxPooling2D(
        (3, 3), strides=(
            1, 1), padding='same')(A_prev)
    maxpool_conv = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu')(maxpool)
    output = K.layers.Concatenate(
        axis=-1)([conv1x1, conv3x3, conv5x5, maxpool_conv])
    return output
