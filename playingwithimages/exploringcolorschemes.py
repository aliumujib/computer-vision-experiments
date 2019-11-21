import cv2
import numpy as np


passport = cv2.imread('ALIU_ABDULMUJEEB_OLOLADE.jpg')

# OpenCV actually stores values in BGR format as opposed RGB
# Print the first pixel
B, G, R = passport[0, 0]
print(B, G, R)
print(passport.shape)

blue, green, red = cv2.split(passport)
# cv2.imshow("blue", blue)
# cv2.imshow("green", green)
# cv2.imshow("red", red)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


enhanced_blue = cv2.merge([blue + 100, green, red])
cv2.imshow("enhanced blue", enhanced_blue)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to black and white
white_black = cv2.cvtColor(passport, cv2.COLOR_BGR2GRAY)
print(white_black.shape)

# HSV
hsv_passport = cv2.cvtColor(passport, cv2.COLOR_BGR2HSV)
print(hsv_passport.shape)


hue_channel = hsv_passport[:, :, 0]
saturation_channel = hsv_passport[:, :, 1]
value_channel = hsv_passport[:, :, 2]


cv2.imshow("Hue", hue_channel)
cv2.imshow("Saturation", saturation_channel)
cv2.imshow("Value", value_channel)

cv2.waitKey(0)
cv2.destroyAllWindows()
