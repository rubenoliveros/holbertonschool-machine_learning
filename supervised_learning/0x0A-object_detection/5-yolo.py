#!/usr/bin/env python3

""" Module that uses the Yolo v3 algorithm to perform
object detection"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2


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
        """ Function that process the Outputs of an image """

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """public method to filter the boxes"""

        scores = [x * y for x, y in zip(box_confidences, box_class_probs)]

        box_class_scores = [np.max(x, axis=-1).reshape(-1) for x in scores]
        box_class_scores = np.concatenate(box_class_scores)

        classes = [np.argmax(x, axis=-1).reshape(-1) for x in scores]
        classes = np.concatenate(classes)

        filtering_mask = box_class_scores >= self.class_t
        list = [np.reshape(x, (-1, 4)) for x in boxes]
        boxes = np.concatenate(list)

        filtered_boxes = boxes[filtering_mask]
        box_scores = box_class_scores[filtering_mask]
        box_classes = classes[filtering_mask]

        return filtered_boxes, box_classes, box_scores

    def MaxMin(self, boxA, boxB):
        """ This is the MaxMin of the box with another one"""

        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])

        interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        MaxMin = interArea / float(boxAArea + boxBArea - interArea)
        return MaxMin

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Method Non Max Suppression"""

        sort_indexes = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[x] for x in sort_indexes])
        predictedClasses = np.array([box_classes[x] for x in sort_indexes])
        predictedScores = np.array([box_scores[x] for x in sort_indexes])

        _, ClassCounts = np.unique(predictedClasses, return_counts=True)
        index = 0
        i = 0

        for n in ClassCounts:
            while i < index + n:
                j = i + 1
                while j < index + n:
                    MaxMinVar = self.MaxMin(
                        box_predictions[i], box_predictions[j])
                    if MaxMinVar > self.nms_t:
                        box_predictions = np.delete(box_predictions, j, axis=0)
                        predictedClasses = np.delete(
                            predictedClasses, j, axis=0)
                        predictedScores = np.delete(predictedScores,
                                                    j, axis=0)
                        n = n - 1
                    else:
                        j = j + 1
                i = i + 1
            index = index + n
        return box_predictions, predictedClasses, predictedScores

    @staticmethod
    def load_images(folder_path):
        """ Static Method that manage images """

        images = []

        image_paths = glob.glob(folder_path + "/*")

        for image in image_paths:
            images.append(cv2.imread(image))

        return images, image_paths

    def preprocess_images(self, images):
        """public method to preprocess images"""

        inputw = self.model.input.shape[1].value
        inputh = self.model.input.shape[2].value

        resize = (inputw, inputh)
        image_shapes = []
        pimages = []

        for image in images:
            shape = image.shape[:2]
            image_shapes.append(shape)
            image_resize = cv2.resize(
                image, resize, interpolation=cv2.INTER_CUBIC)
            image_rescaled = image_resize / 255
            pimages.append(image_rescaled)

        pimages = np.array(pimages)
        image_shapes = np.stack(image_shapes, axis=0)

        return pimages, image_shapes
