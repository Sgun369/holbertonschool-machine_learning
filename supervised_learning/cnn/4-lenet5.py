#!/usr/bin/env python3
"""LeNet-5 (Tensorflow 1)"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5
    architecture
    using tensorflow"""
    conv_lay = tf.layers.conv2d(
        inputs=x, filters=6, kernel_size=(5, 5), padding='same',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0), activation=tf.nn.relu)

    max_pool_lay = tf.layers.max_pooling2d(
        inputs=conv_lay, pool_size=(2, 2), strides=(2, 2))

    conv_lay2 = tf.layers.conv2d(
        inputs=max_pool_lay, filters=16, kernel_size=(5, 5), padding='valid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0), activation=tf.nn.relu)

    max_pool_lay2 = tf.layers.max_pooling2d(
        inputs=conv_lay2, pool_size=(2, 2), strides=(2, 2))

    flat = tf.layers.flatten(inputs=max_pool_lay2)

    fully_con1 = tf.layers.dense(
        inputs=flat,
        units=120,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        activation=tf.nn.relu)

    fully_con2 = tf.layers.dense(
        inputs=fully_con1,
        units=84,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        activation=tf.nn.relu)

    fully_con3 = tf.layers.dense(
        inputs=fully_con2,
        units=10,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))

    y_pred = tf.nn.softmax(fully_con3)
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=fully_con3)
    grad_desc = tf.train.AdamOptimizer().minimize(loss)
    return y_pred, grad_desc, loss, acc
