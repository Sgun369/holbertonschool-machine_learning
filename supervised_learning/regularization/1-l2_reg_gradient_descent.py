#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases
    of a neural network using gradient descent
    with L2 regularization"""
    num_examples = Y.shape[1]

    # Compute delta for the output layer
    delta_output_layer = cache[f'A{L}'] - Y

    for layer in range(L, 0, -1):
        current_weight_key = f'W{layer}'
        current_bias_key = f'b{layer}'
        current_activation_key = f'A{layer}'
        prev_activation_key = f'A{layer-1}' if layer > 1 else 'A0'

        current_activation = cache[current_activation_key]
        prev_activation = cache[prev_activation_key]

        # Calculate gradients for the weights and biases
        weight_gradient = (1 / num_examples) * np.dot(delta_output_layer,
                                                      prev_activation.T) + (lambtha / num_examples) * weights[current_weight_key]
        bias_gradient = (1 / num_examples) * \
            np.sum(delta_output_layer, axis=1, keepdims=True)

        if layer > 1:
            # Compute delta for the previous layer
            weight_matrix = weights[current_weight_key]
            delta_prev_layer = np.dot(weight_matrix.T, delta_output_layer)
            delta_output_layer = delta_prev_layer * \
                (1 - prev_activation ** 2)  # Derivative of tanh

        # Update weights and biases
        weights[current_weight_key] -= alpha * weight_gradient
        weights[current_bias_key] -= alpha * bias_gradient
