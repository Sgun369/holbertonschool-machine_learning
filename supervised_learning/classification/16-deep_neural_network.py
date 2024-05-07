#!/usr/bin/env python3
"""DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        """constructor to initialize
        the deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(layer, int) or layer < 1 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.cache = {}  # intermediary values of the network
        self.weights = {}  # weights and biases of the network
        # loop through each layer to initialize weights and biases
        for i in range(1, self.L + 1):
            weight_key = f"W{i}"
            bias_key = f"b{i}"
            layer_size = layers[i - 1]
            prev_layer_size = nx if i == 1 else layers[i - 2]

        # He and al initialization method for weights
        self.weights[weight_key] = np.random.randn(
            layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
        self.weights[bias_key] = np.zeros((layer_size, 1))
