#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation
    over a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((W_prev - 1) * sw + kw - w_prev) // 2
    elif padding == "valid":
        ph = pw = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    h_out = ((h_prev - kh + 2 * ph) // sh) + 1
    w_out = ((w_prev - kw + 2 * pw) // sw) + 1

    A_prev_padded = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    Z = np.zeros((m, h_out, w_out, c_new))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    A_slice = A_prev_padded[i,
                                            vert_start:vert_end,
                                            horiz_start:horiz_end,
                                            :]
                    Z[i, h, w, c] = np.sum(
                        A_slice * W[:, :, :, c]) + b[:, :, :, c]
    A = activation(Z)
    return A
