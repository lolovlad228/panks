import cv2
import numpy as np
from Classes.Interfase.IClassesification import IClassesificationType
from math import fabs


class ColorClassif(IClassesificationType):

    __low = None
    __high = None

    def img_approximation(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low_color = np.array(self.__low, np.uint8)
        high_color = np.array(self.__high, np.uint8)
        img_filter = cv2.inRange(img_hsv, low_color, high_color)
        img = cv2.bitwise_and(img, img, mask=img_filter)
        cv2.imwrite("img/mask.jpg", img)
        img = cv2.medianBlur(img, 5)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("img/Blur.jpg", img_gray)
        img_filter = cv2.Canny(img_gray, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(img_filter, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("img/close.jpg", closed)
        return closed

    def __init__(self, low, high, color):
        self.__low = low
        self.__high = high
        self.__color = color

    def cycle(self, img):
        contours = cv2.findContours(self.img_approximation(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnt_list = []
        max_area = 0
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            line_len = len(cnt)
            if line_len > 100 and len(approx) >= 8:
                ellipse = cv2.fitEllipse(cnt)
                max_area = line_len if line_len > max_area else max_area
                cnt_list.append((int(ellipse[0][0]), int(ellipse[0][1]), line_len, len(approx), approx))
        squ = list(filter(lambda x: x[3] >= 8 and x[2] == max_area, cnt_list))[-1]
        return [[squ[0], squ[1], f'cycle_{self.__color}', squ[4]]]

    def square(self, img):
        '''contours = cv2.findContours(self.img_approximation(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sqr_list = []
        min_area = 10000000
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            area = w * h / 100
            cv2.drawContours(img, [approx], -1, (255, 0, 0), 4)
            cv2.imwrite("img/test.jpg", img)
            print(area, len(approx))
            if len(approx) == 4 and area > 100:
                min_area = area if min_area > area else min_area
                sqr_list.append((len(approx), w * h / 100, int(x + w / 2), int(y + h / 2)))
        squ = list(filter(lambda x: x[0] == 4 and x[1] == min_area, sqr_list))
        return [[squ[0][2], squ[0][3], 'square']]'''
        img_new = self.img_approximation(img)
        linse = cv2.HoughLinesP(img_new, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
        contours = cv2.findContours(img_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnt_list = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)
            #  print(len(approx), w * h / 100)
            if len(approx) == 4 and 170 >= w * h / 100 > 0.5:
                cnt_list.append([int(x + w / 2), int(y + h / 2), f'square_{self.__color}', approx])
        cnt_list.sort(key=lambda i: i[0] and i[1], reverse=False)
        linse_in_squ = []
        center_list = []
        for i in cnt_list:
            for j in linse:
                x_mid = int((j[0][0] + j[0][2]) / 2)
                y_mid = int((j[0][1] + j[0][3]) / 2)
                if fabs(i[1] - y_mid) < 45 and fabs(i[0] - x_mid) < 45:
                    linse_in_squ.append(j)
            if len(linse_in_squ) == 0:
                center_list.append(i)
                break
            else:
                linse_in_squ.clear()
        return center_list

    def square_with_a_hole(self, img):
        img_new = self.img_approximation(img)
        linse = cv2.HoughLinesP(img_new, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
        contours = cv2.findContours(img_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnt_list = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)
            #  print(len(approx), w * h / 100)
            if len(approx) > 4 and 170 >= w * h / 100 > 0.5:
                cnt_list.append([int(x + w / 2), int(y + h / 2), f'square_with_a_hole_{self.__color}', approx])
        cnt_list.sort(key=lambda i: i[0] and i[1], reverse=False)
        linse_in_squ = []
        center_list = []
        for i in cnt_list:
            for j in linse:
                x_mid = int((j[0][0] + j[0][2]) / 2)
                y_mid = int((j[0][1] + j[0][3]) / 2)
                if fabs(i[1] - y_mid) < 45 and fabs(i[0] - x_mid) < 45:
                    linse_in_squ.append(j)
            if 5 > len(linse_in_squ) > 0:
                center_list.append(i)
                break
        return center_list

