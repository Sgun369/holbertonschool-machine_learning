#!/usr/bin/env python3
""" Privatize Neuron """
import numpy as np


class Neuron:
    """class Neuron that defines a single neuron
    performing binary classification"""

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
