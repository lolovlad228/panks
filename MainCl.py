import cv2
from Classes import Classificate


img_uri = "tuA-YNuPOsE.jpg"

img = cv2.imread(img_uri)
filter_red = Classificate.ColorClassif((50, 60, 185), (10, 255, 255))

cv2.imwrite('aprox.jpg', filter_red.img_approximation(img))

#xy = filter_red.square(img)


#for i in xy:
#    print(i)
#    img = cv2.circle(img, (i[0], i[1]), 100, (255, 255, 255), 2)


cv2.imwrite("now.jpg", img)

