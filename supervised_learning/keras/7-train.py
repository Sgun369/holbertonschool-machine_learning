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

    if early_stopping and validation_data:
        early_stop_callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=patience)
        callbacks.append(early_stop_callback)

    if validation_data and learning_rate_decay:

        def scheduler(epoch):
            """calculates learning rate using inverse time decay"""
            new_lr = alpha / (1 + decay_rate * epoch)
            return new_lr

        lr_schedule_callback = K.callbacks.LearningRateScheduler(
            scheduler, verbose=1)
        callbacks.append(lr_schedule_callback)

        return network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle)
