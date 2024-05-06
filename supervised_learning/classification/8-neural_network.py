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

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = np.zeros((nodes, 1))

        self.W2 = np.random.randn(1, nodes)
        self.b2 = np.zeros((1, 1))
        self.A2 = np.zeros((1, 1))
