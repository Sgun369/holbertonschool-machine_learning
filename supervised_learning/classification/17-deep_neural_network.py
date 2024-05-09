#!/usr/bin/env python3
""" Creating the deep neural network """
import numpy as np


class DeepNeuralNetwork:
    """ deep neural network performing binary classification """

    def __init__(self, nx, layers):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            layer_input_size = nx if i == 0 else layers[i - 1]
            self.__weights['W' + str(i + 1)] = np.random.randn(layers[i],
                                                               layer_input_size) * np.sqrt(2 / layer_input_size)
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter for number of layers. """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache. """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights. """
        return self.__weights
