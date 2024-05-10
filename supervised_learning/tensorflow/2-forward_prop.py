#!/usr/bin/env python3
"""Forward Propagation"""

import tensorflow.compat.v1 as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation
    graph for a neural network"""

    for i in range(len(layer_sizes)):
        output = create_layer(x, layer_sizes[i], activations[i])
        x = output
    return output
