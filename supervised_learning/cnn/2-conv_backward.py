#!/usr/bin/env python3
"""Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a
    convolutional layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        pad_h = 0
        pad_w = 0

    A_prev_padded = np.pad(
        A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
    dA_prev_padded = np.pad(
        dA_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                a_slice = A_prev_padded[:,
                                        vert_start:vert_end,
                                        horiz_start:horiz_end,
                                        :]

                dA_prev_padded[:,
                               vert_start:vert_end,
                               horiz_start:horiz_end,
                               :] += W[:,
                                       :,
                                       :,
                                       k] * dZ[:,
                                               i,
                                               j,
                                               k][:,
                                                  np.newaxis,
                                                  np.newaxis,
                                                  np.newaxis]
                dW[:,
                   :,
                   :,
                   k] += np.sum(a_slice * dZ[:,
                                             i,
                                             j,
                                             k][:,
                                                np.newaxis,
                                                np.newaxis,
                                                np.newaxis],
                                axis=0)

    if padding == "same":
        dA_prev = dA_prev_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
