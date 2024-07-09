#!/usr/bin/env python3
"""Gram Matrix"""
import numpy as np
import tensorflow as tf


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
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        model_outputs = style_outputs + [content_output]

        return tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the gram matrix of an input layer."""

        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")

        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get the dimensions of the input layer
        _, height, width, channels = input_layer.shape

        # Reshape the input layer to a 2D tensor of shape (height * width,
        # channels)
        reshaped_input = tf.reshape(input_layer, (height * width, channels))

        # Calculate the gram matrix
        gram = tf.linalg.matmul(
            reshaped_input,
            reshaped_input,
            transpose_a=True)

        # Normalize the gram matrix by the number of elements in the input
        # layer
        gram /= tf.cast(height * width, tf.float32)

        # Add a batch dimension to the gram matrix
        gram = tf.expand_dims(gram, axis=0)

        return gram
