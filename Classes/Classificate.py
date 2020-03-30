import cv2
import numpy as np
from Classes.Interfase.IClassesification import IClassesificationType


class ColorClassif(IClassesificationType):

    __low = None
    __high = None

    def img_approximation(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.medianBlur(img, 1)
        low_color = np.array(self.__low, np.uint8)
        high_color = np.array(self.__high, np.uint8)
        img_filter = cv2.inRange(img, low_color, high_color)
        return img_filter

    def __init__(self, low, high):
        self.__low = low
        self.__high = high

    def cycle(self, img):
        cycle_filter = cv2.HoughCircles(self.img_approximation(img), cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30,
                                        minRadius=10, maxRadius=100)
        return [i for i in cycle_filter[0, :]]

    def square(self, img):
        contours, _ = cv2.findContours(self.img_approximation(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and w * h > 300:
                cnt_list.append([int(x + w / 2), int(y + h / 2)])
                print(x, y, w, h)
        return cnt_list

    def square_with_a_hole(self, img):
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l2 = clahe.apply(l)

        lab = cv2.merge((l2, a, b))
        mask_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return mask_img