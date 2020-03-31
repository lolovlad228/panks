import cv2
import numpy as np
from Classes.Interfase.IClassesification import IClassesificationType
from math import fabs


class ColorClassif(IClassesificationType):

    __low = None
    __high = None

    def img_approximation(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.medianBlur(img, 5)
        low_color = np.array(self.__low, np.uint8)
        high_color = np.array(self.__high, np.uint8)
        img_filter = cv2.inRange(img, low_color, high_color)
        return img_filter

    def __init__(self, low, high):
        self.__low = low
        self.__high = high

    def cycle(self, img):
        contours, _ = cv2.findContours(self.img_approximation(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_list = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(cnt) > 100 and len(approx) > 10:
                ellipse = cv2.fitEllipse(cnt)
                cnt_list.append([int(ellipse[0][0]), int(ellipse[0][1]), 'cycle'])
        return cnt_list

    def square(self, img):
        contours, _ = cv2.findContours(self.img_approximation(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) < 7 and w * h / 100 > 290:
                cnt_list.append([int(x + w / 2), int(y + h / 2), 'square'])
        return cnt_list

    def square_with_a_hole(self, img):
        contours, _ = cv2.findContours(self.img_approximation(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_list = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)
            if len(approx) < 9 and 290 > w * h / 100 > 0.5:
                cnt_list.append([int(x + w / 2), int(y + h / 2), 'square_with_a_hole'])
        cnt_list.sort(key=lambda i: i[0], reverse=True)
        center_list = []
        for i in range(0, len(cnt_list), 2):
            if fabs(cnt_list[i][0] - cnt_list[i][0]) < 20:
                center_list.append(cnt_list[i])
        return center_list
