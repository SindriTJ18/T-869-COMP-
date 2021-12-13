import cv2
import numpy as np
import time

cap = cv2.VideoCapture('bru laugardagur.mp4')

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


def findObjects(outputs, img, obj_list, car_count_l, car_count_r):
    is_count = False
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for outputs in outputs:
        for det in outputs:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        if ((y+h/2)-650 < 2 and (y+h/2)-650 >= 0):
            is_count = True
            if (x-420 < 0):
                car_count_l += 1
                print(car_count_l)
            else:
                car_count_r += 1
                print(car_count_r)
        plot_boxes(classNames[classIds[i]].upper(), x, y, w, h)
    return obj_list, car_count_l, car_count_r, is_count


def plot_boxes(classId, x, y, w, h):
    if (classId == "CAR"):
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)


def draw_fps(startTime):
    endTime = time.time()
    fps = 1/(endTime - startTime)
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(img, fps_text, (5, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


def plot_ref_line(frame, is_count):
    if (is_count == True):
        cv2.line(frame, (0, 650), (1920, 650), (0, 255, 0), 2)
    else:
        cv2.line(frame, (0, 650), (1920, 650), (0, 0, 255), 2)
    cv2.line(frame, (400, 0), (400, 1080), (0, 0, 255), 2)


def plot_counts(frame, car_count_l, car_count_r):
    cv2.putText(frame,  f'Count left: {car_count_l}', (5, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(frame,  f'Count right: {car_count_r}', (1650, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 30, (1920, 1080))
car_count_l = 0
car_count_r = 0

while True:
    startTime = time.time()
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (widthT, heightT), [0, 0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = []

    for i in (net.getUnconnectedOutLayers() - [1, 1, 1]):
        outputNames.append(layerNames[i])
    outputs = net.forward(outputNames)
    obj_list, car_count_l, car_count_r, is_count = findObjects(
        outputs, img, obj_list, car_count_l, car_count_r)

    # draw_fps(startTime)
    plot_ref_line(img, is_count)
    plot_counts(img, car_count_l, car_count_r)
    out.write(img)
    #cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
