import numpy as np
import dlib
import cv2
import random
import os

print("CWD: {}".format(os.getcwd()))
# Provide the new path here
os.chdir('/Users/aliumujib/Desktop/RoadToAI/OpenCV/facelandmarkdetection/')


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFacesException(Exception):
    pass


class NoFacesException(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFacesException
    if len(rects) == 0:
        raise NoFacesException

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


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


def draw_point(img, p, color):
    cv2.circle(img, p, 2, color=color, thickness=0)

# Draw delaunay triangles


def draw_delaunay(img, subdiv, delaunay_color):

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains_point(r, pt1) and rect_contains_point(r, pt2) and rect_contains_point(r, pt3):

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


# Draw voronoi diagram
def draw_voronoi(img, subdiv):

    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)


image = cv2.imread("ALIU_ABDULMUJEEB_OLOLADE.jpg")
landmarks = get_landmarks(image)

animate = False

delaunay_color = (255, 255, 255)
points_color = (0, 0, 255)

img_orig = image.copy()


size = image.shape
rect = (0, 0, size[1], size[0])

subdiv = cv2.Subdiv2D(rect)

print("Point 1")

for p in landmarks:
    print("Point 2")
    subdiv.insert((p[0, 0], p[0, 1]))
    # Show animation
    if animate:
        img_copy = img_orig.copy()
        # Draw delaunay triangles
        draw_delaunay(img_copy, subdiv, (255, 0, 255))
        cv2.imshow(win_delaunay, img_copy)
        cv2.waitKey(0)


draw_delaunay(image, subdiv, (255, 255, 255))

for p in landmarks:
    draw_point(image, (p[0, 0], p[0, 1]), points_color)

img_voronoi = np.zeros(image.shape, dtype=image.dtype)

draw_voronoi(img_voronoi, subdiv)

cv2.imshow("delaunay trianglation", image)
cv2.imshow("voronoi", img_voronoi)
cv2.waitKey(0)
cv2.destroyAllWindows()
