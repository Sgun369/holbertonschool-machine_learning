#!/usr/bin/env python3
"""Pooling Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    output = np.zeros((m, h_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            for c in range(c_prev):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                if mode == 'max':
                    output[:, 1, j, c] = np.max(
                        A_prev[:, h_start:h_end, w_start:w_end, c], axis=(1, 2))
                elif mode == 'avg':
                    output[:, i, j, c] = np.mean(
                        A_prev[:, h_start:h_end, w_start:w_end, c], axis=(1, 2))
    return output
