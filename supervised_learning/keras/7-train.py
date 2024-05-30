#!/usr/bin/env python3
"""Learning Rate Decay"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        learning_rate_decay=False,
        alpha=0.1,
        decay_rate=1,
        verbose=True,
        shuffle=False):
    """also train the model with learning rate decay"""
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks.append(early_stopping_callback)

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            new_lr = alpha / (1 + decay_rate * epoch)
            print(f"Epoch {epoch + 1}: Learning rate is {new_lr:.6f}")
            return new_lr

        lr_decay_callback = K.callbacks.LearningRateScheduler(
            schedule, verbose=1)
        callbacks.append(lr_decay_callback)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks)
