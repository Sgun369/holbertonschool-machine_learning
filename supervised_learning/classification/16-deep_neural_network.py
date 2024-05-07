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
        for i in range(len(layers)):
            if (not isinstance(layers[i], int)) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                He = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights["W" + str(i + 1)] = He
            else:
                He = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                self.weights['W' + str(i + 1)] = He
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
