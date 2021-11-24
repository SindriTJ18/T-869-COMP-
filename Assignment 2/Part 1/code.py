import cv2
import numpy as np
from numpy import linalg as LA

# GRAB WEBCAM
raw = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def two_random(canny):
    temp = np.argwhere(canny != 0)
    if(len(temp) != 0):
        rand1 = np.random.randint(0, len(temp)-1)
        rand2 = np.random.randint(0, len(temp)-1)
        p1 = temp[rand1]
        p2 = temp[rand2]
        return p1, p2
    else:
        return (0, 0), (0, 0)


def plot_points(canny, p1, p2):
    cv2.circle(canny, (p1[1], p1[0]), 3, (255, 0, 0), -1)
    cv2.circle(canny, (p2[1], p2[0]), 3, (255, 0, 0), -1)


def main():
    while(1):
        # READ WEBCAM AND CROP
        _, frame = raw.read()
        crop_frame = frame[100:480, 0:640]
        # CANNY
        canny = cv2.Canny(crop_frame, 200, 255,
                          apertureSize=3, L2gradient=True)
        edge_points = np.argwhere(canny != 0)
        # FIND TWO RANDOM POINTS
        p1, p2 = two_random(canny)
        for i in range(len(edge_points)):
            p = edge_points[i]
            d = LA.norm(np.cross(p2-p1, p1-edge_points[i]))/LA.norm(p2-p1)
            # print(d)
            if(d < 5):
                cv2.circle(crop_frame, (p[1], p[0]), 1, (0, 0, 255), -1)
        plot_points(crop_frame, p1, p2)
        crop_frame = cv2.resize(crop_frame, (1280, 760))
        cv2.imshow("FRAME", crop_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    raw.release()
    cv2.destroyAllWindows()


main()
