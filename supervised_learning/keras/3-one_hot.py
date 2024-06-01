#!/usr/bin/env python3
"""One Hot"""
import tensorflow.keras as K



def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix"""
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
