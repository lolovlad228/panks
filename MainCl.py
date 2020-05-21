import cv2
from Classes import Classificate
import numpy as np

img_uri = "20191203_190749.jpg"


def nothing(x):
    pass


img = cv2.imread(img_uri)
filter_red = Classificate.ColorClassif((164, 93, 121), (255, 255, 255))
filter_blue = Classificate.ColorClassif((86, 132, 25), (162, 255, 255))
filter_green = Classificate.ColorClassif((70, 93, 42), (95, 255, 255))
filter_list = [filter_red]

list_center = []

def test():
    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LowH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LowS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LowV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UnH", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("UnS", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("UnV", "Tracking", 255, 255, nothing)
    while True:
        frame = cv2.imread(img_uri)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("LowH", "Tracking")
        l_s = cv2.getTrackbarPos("LowS", "Tracking")
        l_v = cv2.getTrackbarPos("LowV", "Tracking")

        h_h = cv2.getTrackbarPos("UnH", "Tracking")
        h_s = cv2.getTrackbarPos("UnS", "Tracking")
        h_v = cv2.getTrackbarPos("UnV", "Tracking")

        l_d = np.array([l_h, l_s, l_v])
        h_d = np.array([h_h, h_s, h_v])
        mask = cv2.inRange(hsv, l_d, h_d)
        mask = cv2.medianBlur(mask, 5)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("REs", res)

        key = cv2.waitKey(1)
        if key == 27:
            break


def main_classificate(img):
    for i in filter_list:
        list_center.append(i.cycle(img))
        list_center.append(i.square(img))
        list_center.append(i.square_with_a_hole(img))
    print(list_center)
    for i in list_center:
        for j in i:
            cv2.circle(img, (j[0], j[1]), 3, (255, 255, 255), 3)
            cv2.drawContours(img, [j[3]], -1, (0, 0, 0), 4)
            cv2.cv2.putText(img, j[2], (j[0] - 60, j[1] - 60), cv2.QT_FONT_NORMAL, 1, (30, 105, 210), 2)
    cv2.imwrite("now.jpg", img)

#  main_classificate(img)
test()

img = cv2.Canny(img, 100, 200)
cv2.imwrite("canny.jpg", img)