import cv2
import csv
import numpy as np
import time
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Render to video
out = cv2.VideoWriter('Video Render.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 30, (1920, 1080))

# Initialize the videocapture object
cap = cv2.VideoCapture('bru_trim.mp4')
input_size = 320

# Detection confidence threshold
confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 0)
font_size = 1
font_thickness = 2

# Middle cross line position
middle_line_position = 650
up_line_position = middle_line_position - 40
down_line_position = middle_line_position + 40

# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]
fps = 0
start_time = 0
end_time = 0
location1 = 0
location2 = 0
start_frames_list = []
end_frames_list = []

# Define speed of vehicles
speed_list = []
start_list = []
end_list = []
measure_list = []
counter = 0
start_measure_list = []
end_measure_list = []
frames = 0
frames_list = []

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]
detected_classNames = []

# Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Function for mesuring speed
def measure_speed(box_id, img):


    x, y, w, h, id, index = box_id
    #print(index)
    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    #print(center)
    global counter
    global location1
    global location2

    if id not in measure_list and iy > topLine(ix) and iy < bottomLine(ix):
        measure_list.append(id)
        start_measure_list.append(id)
        start_list.append(time.time())
        start_frames_list.append(frames)
        
    if id in measure_list and (iy < topLine(ix) or iy > bottomLine(ix)):
        measure_list.remove(id)
        end_measure_list.append(id)
        end_list.append(time.time())
        end_frames_list.append(frames)
        # location1 = start_measure_list.index(id)
        # location2 = end_measure_list.index(id)
        location1 = start_measure_list.index(id)
        location2 = end_measure_list.index(id)
        speed_list.append((14*3.6)/((end_frames_list[location2]-start_frames_list[location1])/30.0))
        
    if len(speed_list) > 0:
        if id in end_measure_list:
            location = end_measure_list.index(id)
            cv2.putText(img, f'{int(speed_list[location])} km/h',
                        (x, y+h+30), cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 2)

# finds center of bounding box
def find_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy

# shows fps count
def FPS(start, img):
    end = time.time()
    frameTime = end - start
    FPS = round(1/frameTime)
    cv2.putText(img, "FPS:  "+str(FPS), (810, 1050),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    # Function for count vehicle

def laneName(cx,cy):
    if (cx > 960):
        cx = cx - 60
        diff = bottomLine(cx) - cy
        cx = cx + (diff)
    else:
        cx = cx + 60
        diff = cy-topLine(cx)
        cx = cx + (diff)
    if (cx < 424):
        lanename = "lane1"
    elif (cx < 537):
        lanename = "lane2"
    elif (cx < 656):
        lanename = "lane3"
    elif (cx < 960):
        lanename = "lane4"
    elif (cx < 1357):
        lanename = "lane5"
    elif (cx < 1515):
        lanename = "lane6"
    else:
        lanename = "lane7"
    return lanename


# Line checking, apply mask on lane
def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    overlay = img.copy()
    if (iy > topLine(ix)) and (iy < bottomLine(ix)):
        plotLaneOverlay(overlay, ix, iy)
    # Find the current position of the vehicle
    if (iy > topLine(ix)) and (iy < middleLine(ix)):
        plotLaneOverlay(overlay, ix, iy)
        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < bottomLine(ix) and iy > middleLine(ix):
        plotLaneOverlay(overlay, ix, iy)
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < topLine(ix):
        if id in temp_down_list:
            temp_down_list.remove(id)
            lane = laneName(ix,iy)
            csvString = "{}, {}, up, {}, {} \n".format(len(speed_list)-2, index, lane, round(speed_list[-1]))
            writeToCSV(csvString)
            up_list[index] = up_list[index] + 1

    elif iy > bottomLine(ix):
        if id in temp_up_list:
            temp_up_list.remove(id)
            lane = laneName(ix,iy)
            csvString = "{}, {}, down, {}, {} \n".format(len(speed_list)-2, index, lane, round(speed_list[-1]))
            writeToCSV(csvString)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the
    img = cv2.addWeighted(overlay, 0.2, img, 1 - 0.2, 0)
    cv2.circle(img, center, 4, (0, 0, 255), -1)  # end here
    return img
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs, img):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w, h = int(det[2]*width), int(det[3]*height)
                    x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(
        boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            detection.append(
                [x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        measure_speed(box_id, img)
        img = count_vehicle(box_id, img)
        
    return img


def writeStatsnew(img):
    # RECTANGLE RED
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (320, 220), (0, 0, 255), -1)
    cv2.rectangle(overlay, (1920, 0), (1600, 220), (0, 0, 255), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0)
    cv2.putText(img, "UP", (1660, 40), cv2.FONT_HERSHEY_DUPLEX,
                font_size, (255, 255, 255), font_thickness)
    cv2.putText(img, "DOWN", (40, 40), cv2.FONT_HERSHEY_DUPLEX,
                font_size, (255, 255, 255), font_thickness)
    # CAR
    cv2.putText(img, "Car:        "+str(up_list[0]), (1660, 80),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Car:        "+str(down_list[0]), (40, 80),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    # MOTORBIKE
    cv2.putText(img, "Motorbike:  "+str(up_list[1]), (1660, 120),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorbike:  "+str(down_list[1]), (40, 120),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    # BUS
    cv2.putText(img, "Bus:        "+str(up_list[2]), (1660, 160),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:        "+str(down_list[2]), (40, 160),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    # TRUCK
    cv2.putText(img, "Truck:      "+str(up_list[3]), (1660, 200),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:      "+str(down_list[3]), (40, 200),
                cv2.FONT_HERSHEY_DUPLEX, font_size, font_color, font_thickness)

    return img
# Function for writing stats to CSV format

# Writes output to CSV file
def writeToCSV(string):
    with open('data.csv', 'a') as the_file:
        the_file.write(string)
        
        #cwriter.writerow(down_list)
    the_file.close()


def realTime():
    while True:
        global frames
        frames +=1
        start = time.time()
        success, img = cap.read()
        blocker = img.copy()
        blocker = drawBlocker(blocker)
        blob = cv2.dnn.blobFromImage(
            blocker, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
        # Draw the crossing section

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = []
        for i in (net.getUnconnectedOutLayers() - [1, 1, 1]):
            outputNames.append(layersNames[i])
        # Feed data to the network
        outputs = net.forward(outputNames)
        # Find the objects from the network output
        img = writeStatsnew(img)
        img = postProcess(outputs, img)
        # Draw counting texts in the frame

        # Show the frames
        FPS(start, img)
        img = plotOverlay(img)
        out.write(img)
        blocker = cv2.resize(blocker, (960, 540))
        cv2.imshow('BONKERS', blocker)

        if cv2.waitKey(1) == ord('q'):
            break
    # Write the vehicle counting information in a file and save it
    #writeToCSV()
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

# Blocks unwanted info for network
def drawBlocker(img):
    cv2.rectangle(img, (0, 0), (1920, 390), (0, 0, 0), -1)
    pt1 = (0, 0)
    pt2 = (0, 500)
    pt3 = (1800, 0)

    ppt1 = (1920, 0)
    ppt2 = (1920, 500)
    ppt3 = (120, 0)

    cv2.circle(img, pt1, 2, (0, 0, 0), -1)
    cv2.circle(img, pt2, 2, (0, 0, 0), -1)
    cv2.circle(img, pt3, 2, (0, 0, 0), -1)

    cv2.circle(img, ppt1, 2, (0, 0, 0), -1)
    cv2.circle(img, ppt2, 2, (0, 0, 0), -1)
    cv2.circle(img, ppt3, 2, (0, 0, 0), -1)

    triangle_cnt = np.array([pt1, pt2, pt3])
    triangle_cnt2 = np.array([ppt1, ppt2, ppt3])
    cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)
    cv2.drawContours(img, [triangle_cnt2], 0, (0, 0, 0), -1)
    return img

# DEF TOPLINE EQ
def topLine(x):
    # 192 down per 960
    if (x < 960):
        line = 570 + x/20
    else:
        line = 618 - (x-960)/20
    return line

# DEF MIDLINE EQ
def middleLine(x):
    # 192 down per 960
    if (x < 960):
        line = 620 + x/10
    else:
        line = 716 - (x-960)/10
    return line

# DEF BOTLINE EQ
def bottomLine(x):
    # 192 down per 960
    if (x < 960):
        line = 670 + x/10
    else:
        line = 766 - (x-960)/10
    return line

# Plots default line masks
def plotOverlay(img):
    overlay = img.copy()
    square_cnt = np.array([(0, 670), (960, 766), (960, 618), (0, 570)])
    square_cnt2 = np.array([(1920, 670), (960, 766), (960, 618), (1920, 570)])
    cv2.drawContours(overlay, [square_cnt], 0, (255, 0, 0), -1)
    cv2.drawContours(overlay, [square_cnt2], 0, (255, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.2, img, 1 - 0.2, 0)
    return img

# Plots lane line masks
def plotLaneOverlay(img, cx, cy):

    lane1 = np.array([(324, 603), (131, 681), (263, 696), (424, 616)])
    lane2 = np.array([(424, 612), (263, 696), (404, 709), (537, 626)])
    lane3 = np.array([(537, 626), (404, 711), (572, 728), (656, 635)])
    lane4 = np.array([(656, 635), (572, 728), (744, 744), (794, 649)])

    lane5 = np.array([(1136, 648), (1192, 742), (1359, 726), (1257, 637)])
    lane6 = np.array([(1257, 637), (1359, 726), (1515, 710), (1383, 624)])
    lane7 = np.array([(1383, 624), (1515, 710), (1666, 694), (1492, 612)])
    if (cx > 960):
        cx = cx - 60
        diff = bottomLine(cx) - cy
        cx = cx + (diff)
    else:
        cx = cx + 60
        diff = cy-topLine(cx)
        cx = cx + (diff)
    if (cx < 424):
        mylane = lane1
        lanename = "lane1"
    elif (cx < 537):
        mylane = lane2
        lanename = "lane2"
    elif (cx < 656):
        mylane = lane3
        lanename = "lane3"
    elif (cx < 960):
        mylane = lane4
        lanename = "lane4"
    elif (cx < 1357):
        mylane = lane5
        lanename = "lane5"
    elif (cx < 1515):
        mylane = lane6
        lanename = "lane6"
    else:
        mylane = lane7
        lanename = "lane7"
    cv2.drawContours(img, [mylane], 0, (0, 255, 0), -1)
    return lanename


if __name__ == '__main__':
    realTime()
