import cv2
import csv
import collections
import numpy as np
#from tracker import *

# Initialize Tracker
#tracker = EuclideanDistTracker()

# Detection confidence threshold
confThreshold = 0.1
nmsThreshold = 0.2

# Middle cross line position
middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# Model Files
modelConfiguration = 'yolov3-slow.cfg'
modelWeights = 'yolov3.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Configure the network backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')
