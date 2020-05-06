import copy
import matplotlib
import math
from skimage.morphology import medial_axis, skeletonize, thin
from skimage import morphology, filters
from skimage.util import invert
import matplotlib.pyplot as plt
import numpy as np
import cv2


def distance(x1, y1, x2, y2):
    square_x = (x2 - x1) ** 2
    square_y = (y2 - y1) ** 2
    return math.sqrt(square_x + square_y)


def opening(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded


def closing(image):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(eroded2, kernel, iterations=1)
    return dilated


image = cv2.imread("cutout1.png")
# image = cv2.imread("/home/hodosy/tmp/pycharm_project_818/cutout1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = image.shape

binary = image > filters.threshold_otsu(image)
np.unique(binary)
# image = np.float32(image)
inv_image = invert(binary)

skeleton = skeletonize(binary)
inv_skeleton = invert(skeleton)



kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(image, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)
eroded2 = cv2.erode(eroded, kernel, iterations=1)
dilated2 = cv2.dilate(eroded2, kernel, iterations=1)

dilated2_binary = dilated2 > filters.threshold_otsu(image)
np.unique(dilated2_binary)

inv_dilated2 = invert(dilated2)

silhouette = dilated2_binary
isObject = False
headFound = False
maxStart = 0
maxEnd = 0
currentStart = 0
index = 0

for i in range(1, height):
    for j in range(1, width):
        if silhouette[i, j] == 1 and silhouette[i, j - 1] == 0:
            isObject = True
            currentStart = j
            headFound = True
        if silhouette[i, j] == 0 and silhouette[i, j - 1] == 1:
            isObject = False
            if j - 1 - currentStart > maxEnd - maxStart:
                maxStart = currentStart
                maxEnd = j - 1
            count = 0
    if headFound:
        break

head_row = i
head_column = int((maxStart + maxEnd) / 2)

isObject = False
maxStart = 0
maxEnd = 0
currentStart = 0
for i in range(1, width):
    if silhouette[int(height / 2), i] == 1 and silhouette[int(height / 2), i - 1] == 0:
        isObject = True
        currentStart = i
    if silhouette[int(height / 2), i] == 0 and silhouette[int(height / 2), i - 1] == 1:
        isObject = False
        if i - 1 - currentStart > maxEnd - maxStart:
            maxStart = currentStart
            maxEnd = i - 1
        count = 0

centroid_row = int(height / 2)
centroid_column = int((maxStart + maxEnd) / 2)

legs = np.zeros((silhouette.shape[0], silhouette.shape[1], 3), dtype="uint8")
legs[centroid_row, centroid_column] = [255, 0, 0]

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
# contours, hierarchy, offset = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

centroid = ([centroid_column, centroid_row])
leftLeg = copy.deepcopy(centroid)
rightLeg = copy.deepcopy(centroid)
for i in range(len(contours)):
    for j in range(0, contours[i].shape[0]):
        if contours[i][j][0][1] > centroid_row:
            # left leg:
            if contours[i][j][0][0] < centroid_column:
                # print("left leg:")
                # print(contours[i][j][0][0])
                # print(centroid)
                # print(distance(centroid[0], centroid[1], contours[i][j][0][0], contours[i][j][0][1]))
                # print(distance(centroid[0], centroid[1], leftLeg[0], leftLeg[1]))
                # print("-------------------------")
                if distance(centroid[0], centroid[1], contours[i][j][0][0], contours[i][j][0][1]) > distance(
                        centroid[0], centroid[1], leftLeg[0], leftLeg[1]):
                    leftLeg[0] = contours[i][j][0][0]
                    leftLeg[1] = contours[i][j][0][1]
            # right leg:
            else:
                # print("right leg:")
                # print(contours[i][j][0][0])
                # print(centroid)
                # print(distance(centroid[0], centroid[1], contours[i][j][0][0], contours[i][j][0][1]))
                # print(distance(centroid[0], centroid[1], leftLeg[0], leftLeg[1]))
                # print("-------------------------")
                if distance(centroid[0], centroid[1], contours[i][j][0][0], contours[i][j][0][1]) > distance(
                        centroid[0], centroid[1], rightLeg[0], rightLeg[1]):
                    rightLeg[0] = contours[i][j][0][0]
                    rightLeg[1] = contours[i][j][0][1]

print(leftLeg)
print(rightLeg)

cv2.line(edgesArray, (leftLeg[0], leftLeg[1]), (centroid[0], centroid[1]), (0, 0, 255), 2)
cv2.line(edgesArray, (rightLeg[0], rightLeg[1]), (centroid[0], centroid[1]), (0, 0, 255), 2)
cv2.line(edgesArray, (head_column, head_row), (centroid[0], centroid[1]), (0, 0, 255), 2)

# cv2.imshow("Edges", edgesArray)

# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

addition = edgesArray

# plotting

fig = plt.figure()

ax = []

ax.append(fig.add_subplot(2, 2, 1))
ax[-1].set_title("Original image")
plt.imshow(inv_image, cmap='Greys', interpolation='nearest')

ax.append(fig.add_subplot(2, 2, 2))
ax[-1].set_title("Filtered image")
plt.imshow(inv_dilated2, cmap='Greys', interpolation='nearest')

ax.append(fig.add_subplot(2, 2, 3))
ax[-1].set_title("First skeleton")
plt.imshow(inv_skeleton, cmap='Greys', interpolation='nearest')

ax.append(fig.add_subplot(2, 2, 4))
ax[-1].set_title("Skeleton corners")
plt.imshow(addition, cmap='Greys', interpolation='nearest')

plt.savefig('skeleton.png')
plt.savefig('skeleton.pdf')

plt.show()
