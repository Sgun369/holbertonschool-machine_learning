#!/usr/bin/env python3
"""Neuron """
import numpy as np


class Neuron:
    """Represent a single neuron for binary classification"""

    def __init__(self, nx):
        """initialize the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx mus be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
