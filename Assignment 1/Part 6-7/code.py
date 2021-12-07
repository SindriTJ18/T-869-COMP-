import cv2
import time
import numpy as np


def plot_box(raw, gray):
    max_loc_x = cv2.minMaxLoc(gray)[3][0]
    max_loc_y = cv2.minMaxLoc(gray)[3][1]
    s = 25
    coord1 = (max_loc_x+s, max_loc_y+s)
    coord2 = (max_loc_x-s, max_loc_y-s)
    color = (255, 100, 0)
    t = 5

    gray_box = np.copy(raw)
    gray_box = cv2.rectangle(gray_box, coord1, coord2, color, t)
    return gray_box


def plot_box_red(raw, hsv, red_mask, gray_box):
    (_, _, _, point) = cv2.minMaxLoc(hsv[:, :, 1], red_mask)

    gray_box = np.copy(gray_box)
    gray_box = cv2.circle(gray_box, point, 8, (0, 0, 255), 5)
    return gray_box


def brightest_for(raw):
    pos = (0, 0)
    brightest = 0
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    (x, y) = gray.shape
    for i in range(x):
        for j in range(y):
            if (gray[i, j] > brightest):
                brightest = gray[i, j]
                pos = (j, i)
    return pos


def main():
    cap = cv2.VideoCapture(
        "http://www.vegagerdin.is/vgdata/vefmyndavelar/myndir/now/artunsbrekka_3.jpg")
    sTime = 0
    fps = 0

    while(1):
        ret, raw = cap.read()
        eTime = time.time()
        fps = 1/(eTime - sTime)

        # DISPLAY FPS TEXT
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(raw, fps_text, (5, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

        # CREATE GRAYSCALE VERSION
        # gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # CREATE HSV VERSION
        # hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        # MASK WITHIN THE RED RANGE
        # low_red = np.array([161, 155, 84])
        # high_red = np.array([179, 255, 255])
        # red_mask = cv2.inRange(hsv, low_red, high_red)
        # LOCATE BRIGHTEST POINT
        # gray_box = plot_box(raw, gray)
        # LOCATE REDDEST POINT
        # gray_box = plot_box_red(raw, hsv, red_mask, gray_box)
        # FIND THEM BRIGHTEST PIXEL WITH DOUBLE FOR LOOP
        pos = brightest_for(raw)
        frame = np.copy(raw)
        # DRAW CIRCLE AROUND BRIGHTEST
        frame = cv2.circle(frame, pos, 8, (0, 0, 255), 5)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sTime = eTime

    cap.release()
    cv2.destroyAllWindows()


main()
