import cv2
import numpy as np
import dlib
import random
import os

os.chdir('/Users/aliumujib/Desktop/RoadToAI/OpenCV/facelandmarkdetection/')
print("CWD: {}".format(os.getcwd()))
# Provide the new path here

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFacesException(Exception):
    pass


class NoFacesException(Exception):
    pass


def rect_contains_point(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Warps and alpha blends triangular regions from img1 and img2 to img
def warp_triangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]                                                      :r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


def calculate_delaunay_triangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    triangle_list = subdiv.getTriangleList()

    delaunay_tri = []

    pt = []

    for t in triangle_list:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains_point(rect, pt1) and rect_contains_point(rect, pt2) and rect_contains_point(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if((abs(pt[j][0] - points[k][0]) < 1.0) and (abs(pt[j][1] - points[k][1]) < 1.0)):
                        # if(np.logical_and((abs(pt[j][0] - points[k][0]) < 1.0), (abs(pt[j][1] - points[k][1]) < 1.0))):
                        ind.append(k)
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunay_tri


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFacesException
    if len(rects) == 0:
        print("No faces found")
        return np.matrix([])

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def elements(array):
    return array.ndim and array.size


def swap_faces(image1, image2):

    image1_warped = np.copy(image2)

    landmarks1 = get_landmarks(image1)
    landmarks2 = get_landmarks(image2)

    if(elements(landmarks1) == 0):
        return np.zeros(image1.shape, image1.dtype)
    elif (elements(landmarks2) == 0):
        return np.zeros(image2.shape, image2.dtype)

    # convert images to float datatype
    image1 = np.float32(image1)
    image2 = np.float32(image2)

    # Find convex hull
    hull1 = []
    hull2 = []

    hull_index = cv2.convexHull(np.array(landmarks2), returnPoints=False)

    for i in range(0, len(hull_index)):
        hull1.append((landmarks1[int(hull_index[i])][0, 0], landmarks1[int(hull_index[i])][0, 1]))
        hull2.append((landmarks2[int(hull_index[i])][0, 0], landmarks2[int(hull_index[i])][0, 1]))

    # Find delanauy traingulation for convex hull points
    sizeImg2 = image2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculate_delaunay_triangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warp_triangle(image1, image1_warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(image2.shape, dtype=image2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(image1_warped), np.uint8(image2),
                               np.uint8(mask), center, cv2.NORMAL_CLONE)
    return output


# Make sure OpenCV is version 3.0 or above
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver) < 3:
    print >>sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
    sys.exit(1)


capture = cv2.VideoCapture(0)

filter_image = cv2.imread("BARRACK_OBAMA.jpg")

while True:
    ret, frame = capture.read()
    frame = cv2.pyrDown(frame)
    cv2.imshow("Original input Swapped", frame)
    output = swap_faces(filter_image, frame)
    output = cv2.pyrDown(output)
    cv2.imshow("Face Swapped", output)
    if cv2.waitKey(1) == 13:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
