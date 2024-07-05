#!/usr/bin/env python3
"""Initialize Initialize"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model


class NST:
    """NST class"""
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image,
                          np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image,
                          np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        max_dim = 512
        h, w, _ = image.shape
        scale = max_dim / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized_image = tf.image.resize(
            image, [new_h, new_w], method=tf.image.ResizeMethod.BICUBIC)
        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)
        return tf.expand_dims(scaled_image, axis=0)

    def load_model(self):
        """creates the model used to calculate cost"""
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        model_outputs = style_outputs + [content_output]

        return Model(vgg.input, model_outputs)
