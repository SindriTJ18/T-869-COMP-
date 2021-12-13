import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)

widthT = 320
heightT = 320

confThreshold = 0.5
nmsThreshold = 0.3
recall = "..."
classesFile = "coco.names"
classNames = []
obj_list = []

# NET CONFIG
net = cv2.dnn.readNet("yolov3.weights", "yolov3-slow.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(classesFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


def findObjects(outputs, img, obj_list):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for outputs in outputs:
        for det in outputs:
            # THE OBJECTS
            scores = det[5:]
            # WHICH ONE IS MOST LIKELY
            classId = np.argmax(scores)
            # HOW CONFIDENT
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    if (len(classIds) == 0):
        obj_list.append(0)
    else:
        obj_list.append(classIds[0])
    # print(obj_list)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return obj_list


def draw_fps(startTime):
    endTime = time.time()
    fps = 1/(endTime - startTime)
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(img, fps_text, (5, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


def draw_recall(recall):
    recall_text = "Precision: {} %".format(recall)
    cv2.putText(img, recall_text, (380, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


while True:
    # START FOR FPS COUNTER
    startTime = time.time()
    # GRAB IMAGE
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (widthT, heightT), [0, 0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = []
    # SHIFT GET OUT LAYERS ONE LEFT
    for i in (net.getUnconnectedOutLayers() - [1, 1, 1]):
        outputNames.append(layerNames[i])
    # FORWARD TO YOLO
    outputs = net.forward(outputNames)
    # FUNC
    obj_list = findObjects(outputs, img, obj_list)
    if (len(obj_list) > 99):
        recall = obj_list.count(41)
        print(recall)
        obj_list = []
    draw_recall(recall)
    draw_fps(startTime)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
