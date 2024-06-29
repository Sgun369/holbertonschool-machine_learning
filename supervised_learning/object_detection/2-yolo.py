#!/usr/bin/env python3
"""Process Outputs
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """In this task, we use the yolo.h5 file."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize class constructor"""
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        with open(classes_path, 'r') as f:
            classes = f.read().strip().split('\n')
        self.class_names = classes

    def sigmoid(self, x):
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process and normalize the output of the YoloV3 model."""
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, nb_box, _ = output.shape
            box_conf = self.sigmoid(output[..., 4:5])
            box_prob = self.sigmoid(output[..., 5:])
            box_confidences.append(box_conf)
            box_class_probs.append(box_prob)

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_x = np.tile(c_x, [grid_h, 1, nb_box])
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_y = np.tile(c_y, [1, grid_w, nb_box])

            p_w = self.anchors[i, :, 0].reshape(1, 1, nb_box)
            p_h = self.anchors[i, :, 1].reshape(1, 1, nb_box)

            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h
            b_w = (np.exp(t_w) * p_w) / self.model.input.shape[1]
            b_h = (np.exp(t_h) * p_h) / self.model.input.shape[2]

            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h

            box = np.zeros((grid_h, grid_w, nb_box, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes based on their box scores"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_class_prob in zip(
                boxes, box_confidences, box_class_probs):
            box_conf = self.sigmoid(box_conf)
            box_class_prob = self.sigmoid(box_class_prob)
            box_scores_raw = box_conf * box_class_prob

            box_classes_raw = np.argmax(box_scores_raw, axis=-1)
            box_scores_raw = np.max(box_scores_raw, axis=-1)

            # Filter out boxes based on scores
            filtering_mask = box_scores_raw >= self.class_t

            filtered_boxes.append(box[filtering_mask])
            box_classes.append(box_classes_raw[filtering_mask])
            box_scores.append(box_scores_raw[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
