import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('ALIU_ABDULMUJEEB_OLOLADE.jpg')

# first agument is the source image, has to be provided in square brackets
# second the channel B[0], G[1], R[2]
# third is mask : to learn what this shit does later
# fourth is the BIN size of the histogram
# fifth is the range, usually [0, 256]

#histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
#plt.hist(image.ravel(), 256, [0, 256])
# plt.show()


# viewing all color channels in one histogram
colors = ['r', 'g', 'b']

for index, color in enumerate(colors):
    histogram2 = cv2.calcHist([image], [index], None, [256], [0, 256])
    plt.plot(histogram2, color=color)
    plt.xlim([0, 256])

plt.show()
