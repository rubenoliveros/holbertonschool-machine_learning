#!/usr/bin/env python3
""" Module that uses the Yolo v3 algorithm to perform
object detection"""

import tensorflow.keras as K


class Yolo:
    """ Class Yolo that uses the Yolo v3 algorithm
        to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):

        with open(classes_path) as f:
            classes_t = f.readlines()

        classes = [x.strip() for x in classes_t]

        self.model = K.models.load_model(model_path)
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
