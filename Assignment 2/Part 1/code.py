import cv2
import numpy as np
from numpy import linalg as LA
from numpy.lib.polynomial import polyfit

# GRAB WEBCAM
raw = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def main():
    while(1):
        # READ WEBCAM AND CROP
        _, frame = raw.read()
        crop_frame = frame[100:480, 0:640]
        # CANNY
        canny = cv2.Canny(crop_frame, 200, 255,
                          apertureSize=3, L2gradient=True)
        edge_points = np.argwhere(canny != 0)

        most_inside = 0
        # ROLL THROUGH 'X' AMOUNT OF POINT PAIRS
        for k in range(30):
            # FIND TWO RANDOM POINTS
            p1, p2 = two_random(canny)
            inside = 0
            # SEARCH FOR INLIERS
            for i in range(0, len(edge_points), 20):
                p = edge_points[i]
                d = LA.norm(np.cross(p2-p1, p1-p))/LA.norm(p2-p1)
                if(d < 4):
                    inside += 1
            # STORE WHICH POINT PAIR HAS MOST INLIERS
            if (inside > most_inside):
                most_inside = inside
                best_p1 = p1
                best_p2 = p2

        x_cords = np.empty(most_inside)
        y_cords = np.empty(most_inside)
        index = 0
        # STORE COORDINATES OF THOSE INLIERS
        for i in range(0, len(edge_points), 20):
            p = edge_points[i]
            d = LA.norm(np.cross(best_p2-best_p1, best_p1 -
                        p))/LA.norm(best_p2-best_p1)
            if(d < 4):
                x_cords[index] = p[0]
                y_cords[index] = p[1]
                index += 1
        # FIND LINE EQUATION
        h, x = polyfit(x_cords, y_cords, 1)
        y1 = int(-1000*h + x)
        y2 = int(1000*h + x)
        # PLOT
        cv2.line(crop_frame, (y1, -1000), (y2, 1000), (0, 255, 0), 2)

        crop_frame = cv2.resize(crop_frame, (1280, 760))
        cv2.imshow("FRAME", crop_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    raw.release()
    cv2.destroyAllWindows()


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


main()
