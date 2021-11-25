import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

# while True:
#     success, img = cap.read()
#     cv2.imshow('Image', img)
#     cv2.waitKey(1)
