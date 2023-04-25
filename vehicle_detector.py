import cv2
import numpy as np


class VehicleDetector:
    def __init__(self):
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1/255)

        self.classes_allowed = [2, 3, 4, 5, 6, 7, 8]

    def detect_vehicles(self, img):
        vehicles_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if class_id in self.classes_allowed and score > 0.7:
                vehicles_boxes.append(box)

        return vehicles_boxes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
