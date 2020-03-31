import cv2
from Classes import Classificate

img_uri = "tuA-YNuPOsE.jpg"

img = cv2.imread(img_uri)
filter_red = Classificate.ColorClassif((0, 50, 150), (10, 255, 255))
filter_blue = Classificate.ColorClassif((94, 80, 2), (126, 255, 255))
filter_green = Classificate.ColorClassif((35, 150, 50), (75, 255, 255))
filter_list = [filter_red, filter_blue, filter_green]

list_center = []


def main_classificate(img):
    for i in filter_list:
        list_center.append(i.cycle(img))
        list_center.append(i.square(img))
        list_center.append(i.square_with_a_hole(img))
    print(list_center)
    for i in list_center:
        for j in i:
            img = cv2.circle(img, (j[0], j[1]), 3, (255, 255, 255), 2)
    cv2.imwrite("now.jpg", img)


main_classificate(img)