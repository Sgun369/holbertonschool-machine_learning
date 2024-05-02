#!/usr/bin/env python3
"""Neuron Forward Propagation"""
import numpy as np


class Neuron:
    """Neuron that defines a single neuron
    performing binary classification """

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights vector for the neuron"""
        return self.__W

    @property
    def b(self):
        """Get the bias for the neuron"""
        return self.__b

    @property
    def A(self):
        """Get the activatted output of the neuron"""
        return self.__A

    def forward_prop(self, X):
        """Perform forward propagation of
        the neuron using sigmoid activation function"""

        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost
