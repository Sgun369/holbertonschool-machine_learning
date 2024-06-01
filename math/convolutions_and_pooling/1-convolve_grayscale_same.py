#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):

            image_slice = padded_images[:, i:i + kh, j:j + kw]

            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
