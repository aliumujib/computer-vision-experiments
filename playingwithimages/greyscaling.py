import cv2
import numpy as np


image = cv2.imread('ALIU_ABDULMUJEEB_OLOLADE.jpg')

grey_scale_photo = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('ALIU_ABDULMUJEEB_OLOLADE_BLACK_WHITE.jpg', grey_scale_photo)


cv2.imshow('BLACK_WHITE', grey_scale_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
