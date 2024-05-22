#!/usr/bin/env python3
"""Create Confusion"""

import numpy as np

def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)
    
    num_classes = labels.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(true_labels, predicted_labels):
        confusion_matrix[true, pred] += 1
    
    return confusion_matrix
