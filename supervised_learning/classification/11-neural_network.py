#!/usr/bin/env python3
"""NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """class NeuralNetwork"""

    def __init__(self, nx, nodes):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integr")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weight vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Bias for the hidden layer"""
        return self.b1

    @property
    def A1(self):
        """Activated output for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Weight vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Activated output of the neuron"""
        return self.__A2

    def sigmoid(self, Z):
        """Sigmoid activation funtion"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Perform forward propagation of the neural network"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model
        using logistic regression"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost
