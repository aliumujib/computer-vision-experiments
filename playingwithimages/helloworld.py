import cv2
import numpy as np

image = cv2.imread('search.png')
# prints shape of the image in width height and RGB
print(image.shape)

cv2.imwrite('copy.png', image)

copy = cv2.imread('copy.png')

cv2.imshow('Hello World copy', copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
