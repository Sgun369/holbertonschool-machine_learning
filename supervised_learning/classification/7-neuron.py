#!/usr/bin/env python3
"""Neuron Forward Propagation"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate on pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dw = np.dot(X, dz.T) / m
        db = np.sum(dz) / m

        self.__W -= alpha * dw.T
        self.__b -= alpha * db

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """Trains the neuron by updating its parameters
        and optionally printing
        and plotting the progress"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step mustt be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if i == 0 or i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print(f"Costt after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(X, Y, alpha)

        if graph:
            plt.plot(range(0, iterations + 1, step), costs)
            plt.xlabel('Iteratons')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
