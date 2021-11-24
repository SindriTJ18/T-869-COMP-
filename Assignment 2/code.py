import cv2
import numpy as np
from numpy import linalg as LA

# GRAB WEBCAM
raw = cv2.VideoCapture(0)


def two_random(canny):
    temp = np.argwhere(canny != 0)
    rand1 = np.random.randint(0, len(temp)-1)
    rand2 = np.random.randint(0, len(temp)-1)
    p1 = temp[rand1]
    p2 = temp[rand2]
    return p1, p2


def plot_points(canny, p1, p2):
    cv2.circle(canny, (p1[1], p1[0]), 3, (255, 0, 0), 2)
    cv2.circle(canny, (p2[1], p2[0]), 3, (255, 0, 0), 2)
    ...


def main():
    while(1):
        # READ WEBCAM AND CROP
        _, frame = raw.read()
        crop_frame = frame[100:480, 0:640]
        # CANNY
        canny = cv2.Canny(crop_frame, 200, 255,
                          apertureSize=3, L2gradient=True)
        temp = np.argwhere(canny != 0)
        # FIND TWO RANDOM POINTS
        p1, p2 = two_random(canny)
        plot_points(crop_frame, p1, p2)
        for j in range(3):
            for i in range(len(temp)):
                p = temp[i]
                d = LA.norm(np.cross(p2-p1, p1-temp[i]))/LA.norm(p2-p1)
                # print(d)
                if(d < 10):
                    cv2.circle(crop_frame, (p[1], p[0]), 1, (255, 255, 255), 1)
        cv2.imshow("FRAME", canny)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break
    raw.release()
    cv2.destroyAllWindows()


main()
