from dataclasses import dataclass
from skimage import morphology, filters
from skimage.util import invert

import copy
import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2


@dataclass
class Coordinates:
    column: int
    row: int

    def __init__(self, column, row):
        self.column = column
        self.row = row

    @property
    def column(self) -> int:
        return self.column

    def row(self) -> int:
        return self.row


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


def find_head(img, img_height, img_width):
    isObject = False
    headFound = False
    maxStart = 0
    maxEnd = 0
    currentStart = 0
    index = 0

    for i in range(1, img_height):
        for j in range(1, img_width):
            if img[i, j] == 1 and img[i, j - 1] == 0:
                isObject = True
                currentStart = j
                headFound = True
            if img[i, j] == 0 and img[i, j - 1] == 1:
                isObject = False
                if j - 1 - currentStart > maxEnd - maxStart:
                    maxStart = currentStart
                    maxEnd = j - 1
                count = 0
        if headFound:
            break

    head_row = i
    head_column = int((maxStart + maxEnd) / 2)

    head_coordinates = Coordinates(head_column, head_row)

    return head_coordinates


def find_centroid(img, img_height, img_width):
    isObject = False
    maxStart = 0
    maxEnd = 0
    currentStart = 0
    for i in range(1, img_width):
        if img[int(img_height / 2), i] == 1 and img[int(img_height / 2), i - 1] == 0:
            isObject = True
            currentStart = i
        if img[int(img_height / 2), i] == 0 and img[int(img_height / 2), i - 1] == 1:
            isObject = False
            if i - 1 - currentStart > maxEnd - maxStart:
                maxStart = currentStart
                maxEnd = i - 1
            count = 0

    centroid_row = int(height / 2)
    centroid_column = int((maxStart + maxEnd) / 2)

    centroid_coordinates = Coordinates(centroid_column, centroid_row)
    return centroid_coordinates


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
    ax[-1].set_title("First skeleton")
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

legs = np.zeros((silhouette.shape[0], silhouette.shape[1], 3), dtype="uint8")
legs[centroid.row, centroid.column] = [255, 0, 0]

# skeleton_final = skeletonize(dilated2_binary)
# inv_skeleton_final = invert(skeleton_final)
silhouette = np.uint8(silhouette)

v = np.median(silhouette)
sigma = 0.33

lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

edges = cv2.Canny(silhouette, lower, upper)

edgesArray = np.zeros((edges.shape[0], edges.shape[1], 3))
edgesArray[edges == 255] = [255, 255, 255]

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

leftLeg = copy.deepcopy(centroid)
rightLeg = copy.deepcopy(centroid)
for i in range(len(contours)):
    for j in range(0, contours[i].shape[0]):
        if contours[i][j][0][1] > centroid.row:
            # left leg:
            if contours[i][j][0][0] < centroid.column:
                if distance(centroid.column, centroid.row, contours[i][j][0][0], contours[i][j][0][1]) > distance(
                        centroid.column, centroid.row, leftLeg[0], leftLeg[1]):
                    leftLeg[0] = contours[i][j][0][0]
                    leftLeg[1] = contours[i][j][0][1]
            # right leg:
            else:
                if distance(centroid.column, centroid.row, contours[i][j][0][0], contours[i][j][0][1]) > distance(
                        centroid.column, centroid.row, rightLeg[0], rightLeg[1]):
                    rightLeg[0] = contours[i][j][0][0]
                    rightLeg[1] = contours[i][j][0][1]

print(leftLeg)
print(rightLeg)

cv2.line(edgesArray, (leftLeg[0], leftLeg[1]), (centroid.column, centroid.row), (0, 0, 255), 2)
cv2.line(edgesArray, (rightLeg[0], rightLeg[1]), (centroid.column, centroid.row), (0, 0, 255), 2)
cv2.line(edgesArray, (head.column, head.row), (centroid.column, centroid.row), (0, 0, 255), 2)

# cv2.imshow("Edges", edgesArray)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# plotting
plot_figures(invert(original_image_binary), invert(image), invert(image), edgesArray)
