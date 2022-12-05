#!/usr/bin/env python3

""" Module that uses the Yolo v3 algorithm to perform
object detection"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Class Yolo that uses the Yolo v3 algorithm
        to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):

        with open(classes_path) as f:
            classes_t = f.readlines()

        # Strip each line to get the Word (Class)
        classes = [x.strip() for x in classes_t]

        self.model = K.models.load_model(model_path)
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ Function that process the Outputs of an image"""

        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        InHeights = self.model.input.shape[2].value
        InWidths = self.model.input.shape[1].value

        for ItemOutput in outputs:

            boxes.append(ItemOutput[..., :4])

            box_confidence = 1 / (1 + np.exp(-(ItemOutput[..., 4:5])))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-(ItemOutput[..., 5:])))
            box_class_probs.append(box_class_prob)

        for i, box in enumerate(boxes):
            grid_height = box.shape[0]
            grid_width = box.shape[1]
            anchor_boxes = box.shape[2]

            HeightMatrix = np.arange(grid_height).reshape(1, grid_height)
            HeightMatrix = np.repeat(HeightMatrix, grid_width, axis=0).T
            HeightMatrix = np.repeat(
                HeightMatrix[:, :, np.newaxis], anchor_boxes, axis=2)
            WidthMatrix = np.arange(grid_width).reshape(1, grid_width)
            WidthMatrix = np.repeat(WidthMatrix, grid_height, axis=0)
            WidthMatrix = np.repeat(
                WidthMatrix[:, :, np.newaxis], anchor_boxes, axis=2)

            box[..., :2] = 1 / (1 + np.exp(-(box[..., :2])))
            box[..., 0] += WidthMatrix
            box[..., 1] += HeightMatrix

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            box[..., 2:] = np.exp(box[..., 2:])
            box[..., 2] *= anchor_width
            box[..., 3] *= anchor_height

            box[..., 0] *= image_width / grid_width
            box[..., 1] *= image_height / grid_height
            box[..., 2] *= image_width / InWidths
            box[..., 3] *= image_height / InHeights

            box[..., 0] -= box[..., 2] / 2
            box[..., 1] -= box[..., 3] / 2
            box[..., 2] += box[..., 0]
            box[..., 3] += box[..., 1]

        return boxes, box_confidences, box_class_probs
