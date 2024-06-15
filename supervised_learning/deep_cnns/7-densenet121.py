#!/usr/bin/env python3
"""DenseNet-121"""
from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in the paper."""
    he_normal = K.initializers.HeNormal(seed=0)

    # Input layer
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial convolution and pooling
    X = K.layers.Conv2D(64, (7, 7), strides=2, padding='same',
                        kernel_initializer=he_normal)(input_layer)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.MaxPooling2D((3, 3), strides=2, padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 64, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final layers
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    output_layer = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=he_normal)(X)

    # Create model
    model = K.Model(inputs=input_layer, outputs=output_layer)

    return model
