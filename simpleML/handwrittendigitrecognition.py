import cv2
import numpy as np
import os

os.chdir('/Users/aliumujib/Desktop/RoadToAI/OpenCV/simpleML/')


def x_contour_cord(contour):
    area = cv2.contourArea(contour)
    if(area > 10):
        M = cv2.moments(area)
        return (int(M['m10'])/int(M['m00']))


def make_square(not_square):
    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    print("Pre squaring: {}x{}".format(width, height))
    if(height == width):
        square = not_square
        return square
    else:
        height = height*2
        width = width*2
        double_size = cv2.resize(not_square, (height, width), interpolation=cv2.INTER_CUBIC)
        print("Pre squaring after doublees: {}x{}".format(width, height))
        if(height > width):
            pad = int((height - width)/2)
            print("Padding sides: {}".format(pad))
            double_size_square = cv2.copyMakeBorder(
                double_size, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = int((width - height)/2)
            print("Padding top down: {}".format(pad))
            double_size_square = cv2.copyMakeBorder(
                double_size, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=BLACK)
    double_square_dim = double_size_square.shape
    print("Post squaring: {}x{}".format(double_square_dim[0], double_square_dim[1]))
    return double_size_square


def resize_to_pixel(dimensions, image_to_reshape):
    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image_to_reshape
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image_to_reshape, dim, interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if(height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=BLACK)
    if(height_r < width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    p = 2
    resized_image = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)
    print("Shape of resized image {}:".format(resized_image.shape))
    return resized_image


image = cv2.imread("digits.png", 0)
cv2.imshow("digits", image)
cv2.waitKey(0)

cell = [np.hsplit(row, 100) for row in np.vsplit(image, 50)]

data = np.array(cell)

print("data Shape: {}".format(data.shape))

train = data[:, :50].reshape(-1, 400).astype(np.float32)
test = data[:, 50:100].reshape(-1, 400).astype(np.float32)

print("train Shape: {}".format(train.shape))
print("test Shape: {}".format(test.shape))

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, distance = knn.findNearest(test, k=5)

matches = result = test_labels
correct = np.count_nonzero(matches)
accuracy = correct * (100/result.size)
print("Accuracy is {}:".format(accuracy))


image = cv2.imread("numbers.jpg")
image = cv2.pyrDown(image)
image = cv2.pyrDown(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", image)
cv2.waitKey(0)

#blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(gray, 30, 150)
cv2.imshow("edged", edged)
cv2.waitKey(0)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print("type: {}".format(contours))

#contours = sorted(contours, key=x_contour_cord, reverse=False)

full_number = []

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    if w >= 5 and h >= 25:
        roi = edged[y:y+h, x:x+w]
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        squared = make_square(roi)
        final = resize_to_pixel(20, squared)
        cv2.imshow("final", final)
        # cv2.waitKey(0)
        print("Shape of final: {}".format(final.shape))
        final_array = final.reshape((-1, 400)).astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(final_array, k=5)
        print("Results {}:".format(result))
        number = str(int(float(result[0])))
        full_number.append(number)

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, number, (x, y+155), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)


cv2.destroyAllWindows()
print("The number is: "+''.join(full_number))
