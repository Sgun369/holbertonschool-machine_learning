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
        self.__cache = dict()
        self.__weights = dict()
        for i in range(len(layers)):
            if (not isinstance(layers[i], int)) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                He = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(i + 1)] = He
            else:
                He = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                self.__weights['W' + str(i + 1)] = He
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Return layers in the deep neural network """
        return self.__L

    @property
    def cache(self):
        """ Return the values stored in cache """
        return self.__cache

    @property
    def weights(self):
        """ Return the values stored in the weights dictionary """
        return self.__weights
