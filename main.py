from dataclasses import dataclass
from skimage import filters
from skimage.util import invert

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2


@dataclass
class Coordinates:
    def __init__(self, column, row):
        self._column = column
        self._row = row

    @property
    def column(self) -> int:
        return self._column

    @property
    def row(self) -> int:
        return self._row

    @column.setter
    def column(self, i: int) -> None:
        self._column = i

    @row.setter
    def row(self, i: int) -> None:
        self._row = i


def img_read(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binarize(img):
    binary = img > filters.threshold_otsu(img)
    np.unique(binary)
    return binary


def opening(img, iterations):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=iterations)
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)
    return dilated


def closing(img, iterations):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    eroded = cv2.erode(dilated, kernel, iterations=iterations)
    return eroded


def distance(x1, y1, x2, y2):
    square_x = (x2 - x1) ** 2
    square_y = (y2 - y1) ** 2
    return math.sqrt(square_x + square_y)


def contour_farther_from_centroid(contour_point, current_leg_point, centroid):
    distance1 = distance(centroid.column, centroid.row, contour_point[0], contour_point[1])
    distance2 = distance(centroid.column, centroid.row, current_leg_point.column, current_leg_point.row)
    return distance1 > distance2


def Canny_edges(img):
    img = np.uint8(img)

    v = np.median(img)
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edge = cv2.Canny(img, lower, upper)
    return edge


def edges_to_array(edge):
    array = np.zeros((edge.shape[0], edges.shape[1], 3))
    array[edges == 255] = [255, 255, 255]
    return array


def find_head(img, img_height, img_width):
    head_found = False
    max_start = 0
    max_end = 0
    current_start = 0

    for i in range(1, img_height):
        for j in range(1, img_width):
            if img[i, j] == 1 and img[i, j - 1] == 0:
                current_start = j
                head_found = True
            if img[i, j] == 0 and img[i, j - 1] == 1:
                if j - 1 - current_start > max_end - max_start:
                    max_start = current_start
                    max_end = j - 1
        if head_found:
            break

    head_row = i
    head_column = int((max_start + max_end) / 2)

    head_coordinates = Coordinates(head_column, head_row)

    return head_coordinates


def find_centroid(img, img_height, img_width):
    max_start = 0
    max_end = 0
    current_start = 0
    for i in range(1, img_width):
        if img[int(img_height / 2), i] == 1 and img[int(img_height / 2), i - 1] == 0:
            current_start = i
        if img[int(img_height / 2), i] == 0 and img[int(img_height / 2), i - 1] == 1:
            if i - 1 - current_start > max_end - max_start:
                max_start = current_start
                max_end = i - 1

    centroid_row = int(height / 2)
    centroid_column = int((max_start + max_end) / 2)

    centroid_coordinates = Coordinates(centroid_column, centroid_row)
    return centroid_coordinates


def find_legs(edge, centroid):
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    left_leg = copy.deepcopy(centroid)
    right_leg = copy.deepcopy(centroid)
    for i in range(len(contours)):
        for j in range(0, contours[i].shape[0]):
            if contours[i][j][0][1] > centroid.row:
                # left leg:
                if contours[i][j][0][0] < centroid.column:
                    if contour_farther_from_centroid(contours[i][j][0], left_leg, centroid):
                        left_leg.column = contours[i][j][0][0]
                        left_leg.row = contours[i][j][0][1]
                # right leg:
                else:
                    if contour_farther_from_centroid(contours[i][j][0], right_leg, centroid):
                        right_leg.column = contours[i][j][0][0]
                        right_leg.row = contours[i][j][0][1]

    return left_leg, right_leg


def draw_skeleton(edge_array, left, right, head, centroid):
    cv2.line(edge_array, (left.column, left.row), (centroid.column, centroid.row), (255, 0, 0), 2)
    cv2.line(edge_array, (right.column, right.row), (centroid.column, centroid.row), (255, 0, 0), 2)
    cv2.line(edge_array, (head.column, head.row), (centroid.column, centroid.row), (255, 0, 0), 2)
    return edge_array


def plot_figures(image1, image2, image3, image4):
    fig = plt.figure()

    ax = []

    ax.append(fig.add_subplot(2, 2, 1))
    ax[-1].set_title("Original image")
    plt.imshow(image1, cmap='Greys', interpolation='nearest')

    ax.append(fig.add_subplot(2, 2, 2))
    ax[-1].set_title("Filtered image")
    plt.imshow(image2, cmap='Greys', interpolation='nearest')

    ax.append(fig.add_subplot(2, 2, 3))
    ax[-1].set_title("Canny edges")
    plt.imshow(image3, cmap='Greys', interpolation='nearest')

    ax.append(fig.add_subplot(2, 2, 4))
    ax[-1].set_title("Skeleton corners")
    plt.imshow(image4, cmap='Greys', interpolation='nearest')

    plt.savefig('skeleton.png')
    plt.savefig('skeleton.pdf')

    plt.show()


original_image = img_read("cutout1.png")

height, width = original_image.shape

original_image_binary = binarize(original_image)

# skeleton = skeletonize(original_image_binary)
# inv_skeleton = invert(skeleton)

image = opening(original_image, 1)
image = closing(image, 1)

silhouette = binarize(image)

head = find_head(silhouette, height, width)
centroid = find_centroid(silhouette, height, width)

edges = Canny_edges(silhouette)
edges_array = edges_to_array(edges)
contours = copy.deepcopy(edges_array)

left_leg, right_leg = find_legs(edges, centroid)

skeleton = draw_skeleton(edges_array, left_leg, right_leg, head, centroid)

# plotting
plot_figures(invert(original_image_binary), invert(image), contours.astype(np.uint8), skeleton.astype(np.uint8))
