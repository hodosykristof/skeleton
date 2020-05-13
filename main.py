from dataclasses import dataclass
from skimage import filters
from skimage.morphology import skeletonize
from skimage.util import invert

import copy
import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import time

input_num = "2780"
filename = input_num + "_bw.png"
source = "input/" + filename


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


def select_correct_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    correct = []

    for i in range(0, len(contours)):
        if len(contours[i]) > 75:
            x = contours[i][0][0][0]
            y = contours[i][0][0][1]
            if 6 * x - 22 * y + 9700 > 0 > 7 * x + 24 * y - 45300:
                correct.append(contours[i])
    return correct


def draw_player_contours(cntrs):
    players = np.zeros([height, width, 3], dtype=np.uint8)
    players.fill(0)
    cv2.drawContours(players, cntrs, -1, (255, 255, 255), 1)
    return players


def Canny_edges(img):
    img = np.uint8(img)

    v = np.median(img)
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edge = cv2.Canny(img, lower, upper)
    return edge


def edges_to_array(edge):
    array = np.zeros((edge.shape[0], edge.shape[1], 3))
    array[edge == 255] = [255, 255, 255]
    array = np.uint8(array)
    return array


def ROI_to_array(img):
    array = np.zeros((img.shape[0], img.shape[1], 3))
    array[img == 1] = [255, 255, 255]
    array = np.uint8(array)
    return array


def find_skeleton_endpoints(img, w, h):
    neighbours = 0
    endpoints = []
    intersections = []

    for i in range(0, h):
        for j in range(0, w):
            if img[i][j]:
                if j != 0:
                    if i != 0:
                        if img[i - 1][j - 1]:
                            neighbours += 1
                    if i != h - 1:
                        if img[i + 1][j - 1]:
                            neighbours += 1
                    if img[i][j - 1]:
                        neighbours += 1
                if i != 0:
                    if img[i - 1][j]:
                        neighbours += 1
                if i != h - 1:
                    if img[i + 1][j]:
                        neighbours += 1
                if j != w - 1:
                    if i != 0:
                        if img[i - 1][j + 1]:
                            neighbours += 1
                    if i != h - 1:
                        if img[i + 1][j + 1]:
                            neighbours += 1
                    if img[i][j + 1]:
                        neighbours += 1
            if neighbours == 1:
                endpoints.append(Coordinates(j, i))
            if neighbours > 2:
                intersections.append(Coordinates(j, i))
            # print(neighbours)
            neighbours = 0

    return endpoints, intersections


def find_head(img, img_height, img_width):
    head_found = False
    max_start = 0
    max_end = 0
    current_start = 0

    for i in range(0, img_height):
        for j in range(0, img_width):
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

    if max_end == 0:
        max_start = current_start
        max_end = i - 1

    centroid_row = int(img_height / 2)
    centroid_column = int((max_start + max_end) / 2)

    centroid_coordinates = Coordinates(centroid_column, centroid_row)
    return centroid_coordinates


def find_shoulders(head, centroid):
    column_dist = head.column - centroid.column
    row_dist = head.row - centroid.row
    column_offset = int(column_dist * 11 / 16)
    row_offset = int(row_dist * 11 / 16)
    shoulder_column = centroid.column + column_offset
    shoulder_row = centroid.row + row_offset

    shoulder_coordinates = Coordinates(shoulder_column, shoulder_row)
    return shoulder_coordinates


def find_left_hand(edge, shoulders, centroid):
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    left_hand = copy.deepcopy(shoulders)
    shoulders_centroid_dist = distance(shoulders.column, shoulders.row, centroid.column, centroid.row)
    shoulders_hand_dist = int(shoulders_centroid_dist * 3.75 / 2.75)
    for i in range(0, contours[0].shape[0]):
        if contours[0][i][0][0] < shoulders.column and abs(
                distance(contours[0][i][0][0], contours[0][i][0][1], shoulders.column,
                         shoulders.row) - shoulders_hand_dist) < abs(
            distance(left_hand.column, left_hand.row, shoulders.column, shoulders.row) - shoulders_hand_dist):
            left_hand.column = contours[0][i][0][0]
            left_hand.row = contours[0][i][0][1]

    return left_hand


def find_right_hand(edge, shoulders, centroid):
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    right_hand = copy.deepcopy(shoulders)
    shoulders_centroid_dist = distance(shoulders.column, shoulders.row, centroid.column, centroid.row)
    shoulders_hand_dist = int(shoulders_centroid_dist * 3.75 / 2.75)
    for i in range(0, contours[0].shape[0]):
        if contours[0][i][0][0] > shoulders.column and abs(
                distance(contours[0][i][0][0], contours[0][i][0][1], shoulders.column,
                         shoulders.row) - shoulders_hand_dist) < abs(
            distance(right_hand.column, right_hand.row, shoulders.column, shoulders.row) - shoulders_hand_dist):
            right_hand.column = contours[0][i][0][0]
            right_hand.row = contours[0][i][0][1]

    return right_hand


def find_legs(edge, centroid):
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    left_leg = copy.deepcopy(centroid)
    right_leg = copy.deepcopy(centroid)
    for i in range(0, contours[0].shape[0]):
        if contours[0][i][0][1] > centroid.row:
            # left leg:
            if contours[0][i][0][0] < centroid.column:
                if contour_farther_from_centroid(contours[0][i][0], left_leg, centroid):
                    left_leg.column = contours[0][i][0][0]
                    left_leg.row = contours[0][i][0][1]
            # right leg:
            else:
                if contour_farther_from_centroid(contours[0][i][0], right_leg, centroid):
                    right_leg.column = contours[0][i][0][0]
                    right_leg.row = contours[0][i][0][1]

    return left_leg, right_leg


def export_legs(legs):
    data = {}
    i = 1
    for l in legs:
        data['player ' + str(i)] = []
        data['player ' + str(i)].append({
            'name': 'left_leg',
            'x_coordinate': int(l[0].column),
            'y_coordinate': int(l[0].row)
        })
        data['player ' + str(i)].append({
            'name': 'right_leg',
            'x_coordinate': int(l[1].column),
            'y_coordinate': int(l[1].row)
        })
        i += 1

    with open('feature_points.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)


def offset_points(point, offset):
    point.row += offset.row
    point.column += offset.column
    return point


def is_below_line(l1, l2, p):
    m = (l2.row - l1.row) / (l2.column - l1.column)
    b = l1.row - m * l1.column
    return p.row > m * p.column + b


def draw_skeleton(edge_array, left, right, head, centroid, shoulders, hand1, hand2):
    cv2.line(edge_array, (left.column, left.row), (centroid.column, centroid.row), (255, 0, 0), 2)
    cv2.line(edge_array, (right.column, right.row), (centroid.column, centroid.row), (255, 0, 0), 2)
    cv2.line(edge_array, (head.column, head.row), (centroid.column, centroid.row), (255, 0, 0), 2)
    cv2.line(edge_array, (hand1.column, hand1.row), (shoulders.column, shoulders.row), (255, 0, 0), 2)
    cv2.line(edge_array, (hand2.column, hand2.row), (shoulders.column, shoulders.row), (255, 0, 0), 2)


def filter_ROI(img, contour, w, h, x, y):
    for i in range(0, h):
        for j in range(0, w):
            if img[i][j]:
                if cv2.pointPolygonTest(contour, (j + x, i + y), False) == -1:
                    img[i][j] = False

    return img


def find_correct_endpoints(ends, intersects, w, h, centroid):
    good_ends = []
    for e in ends:
        small = False
        for i in intersects:
            if 0 < distance(e.column, e.row, i.column, i.row) < h * w / 2000:
                small = True
                intersects.remove(i)
                break
        if e.row > centroid.row and small is False:
            good_ends.append(e)

    for i in intersects:
        small = False
        for e in ends:
            if 0 < distance(e.column, e.row, i.column, i.row) < h * w / 2000:
                small = True
                break
        if i.row > centroid.row and small is False:
            good_ends.append(i)

    return good_ends


def find_legs2(good_ends, ROI_mod, centroid):
    leg1 = copy.deepcopy(centroid)
    leg2 = copy.deepcopy(centroid)
    for e in good_ends:
        if len(good_ends) < 2:
            leg1, leg2 = find_legs(ROI_mod, copy.deepcopy(centroid))
        else:
            if distance(e.column, e.row, centroid.column, centroid.row) > distance(leg2.column, leg2.row,
                                                                                   centroid.column, centroid.row):
                if distance(e.column, e.row, centroid.column, centroid.row) > distance(leg1.column, leg1.row,
                                                                                       centroid.column,
                                                                                       centroid.row):
                    leg2.row = leg1.row
                    leg2.column = leg1.column
                    leg1.row = e.row
                    leg1.column = e.column
                else:
                    leg2.row = e.row
                    leg2.column = e.column

    return leg1, leg2


def draw_all_skeletons(player_contours, centroid, head, leg1, leg2, shoulders, hand1, hand2, offset):
    centroid_offset = offset_points(centroid, offset)
    head_offset = offset_points(head, offset)
    leg1_offset = offset_points(leg1, offset)
    leg2_offset = offset_points(leg2, offset)
    shoulders_offset = offset_points(shoulders, offset)
    hand1_offset = offset_points(hand1, offset)
    hand2_offset = offset_points(hand2, offset)

    draw_skeleton(player_contours, leg1_offset, leg2_offset, head_offset, centroid_offset, shoulders_offset, hand1_offset, hand2_offset)

    return player_contours


def find_players(cntrs, player_contours):
    ROI_number = 0
    hands = []
    centroids = []
    shoulderss = []
    legs = []

    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        offset = Coordinates(x, y)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        ROI = silhouette[y:y + h, x:x + w]
        # cv2.imwrite('ROI_prototype_{}.png'.format(ROI_number), ROI_array)
        ROI_number += 1

        ROI = filter_ROI(ROI, c, w, h, x, y)

        head = find_head(ROI, h, w)
        centroid = find_centroid(ROI, h, w)

        ROI_mod = ROI * np.uint8(255)
        ROI_array = ROI_to_array(ROI)
        ROI_skeleton = skeletonize(ROI_array)
        ROI_skeleton_bin = cv2.cvtColor(ROI_skeleton, cv2.COLOR_BGR2GRAY)
        ROI_skeleton_bin = binarize(ROI_skeleton_bin)
        ends, intersects = find_skeleton_endpoints(ROI_skeleton_bin, w, h)

        correct_endpoints = find_correct_endpoints(ends, intersects, w, h, centroid)

        leg1, leg2 = find_legs2(correct_endpoints, ROI_mod, centroid)

        shoulders = find_shoulders(head, centroid)

        # print(ROI_number)
        possible_hands = []
        for e in ends:
            small = False
            for i in intersects:
                if 0 < distance(e.column, e.row, i.column, i.row) < h * w / 800:
                    small = True
                    break
            intersects = [i for i in intersects if not 0 < distance(e.column, e.row, i.column, i.row) < h * w / 800]
            if small is False and not (e.row == leg1.row and e.column == leg1.column) \
                    and not (e.row == leg2.row and e.column == leg2.column) \
                    and distance(e.column, e.row, leg1.column, leg1.row) > h * w / 350 \
                    and distance(e.column, e.row, leg2.column, leg2.row) > h * w / 350 \
                    and distance(e.column, e.row, head.column, head.row) > h * w / 380 \
                    and e.row <= centroid.row:
                possible_hands.append(e)
                # print(e.column, ";", e.row)

        if len(possible_hands) >= 2:
            hand1 = Coordinates(possible_hands[0].column, possible_hands[0].row)
            hand2 = Coordinates(possible_hands[1].column, possible_hands[1].row)

        elif len(possible_hands) == 1:
            hand1 = Coordinates(possible_hands[0].column, possible_hands[0].row)
            if possible_hands[0].column < shoulders.column:
                hand2 = find_right_hand(ROI_mod, shoulders, centroid)
            else:
                hand2 = find_left_hand(ROI_mod, shoulders, centroid)

        else:
            hand1 = find_right_hand(ROI_mod, shoulders, centroid)
            hand2 = find_left_hand(ROI_mod, shoulders, centroid)
        # print("-----------------------------------------------------------------")

        draw_skeleton(ROI_array, leg1, leg2, head, centroid, shoulders, hand1, hand2)
        cv2.imwrite("output/ROI_new_{}.png".format(ROI_number), ROI_array)

        draw_all_skeletons(player_contours, centroid, head, leg1, leg2, shoulders, hand1, hand2, offset)
        hands.append([hand1, hand2])
        shoulderss.append(shoulders)
        centroids.append(centroid)
        legs.append([leg1, leg2])


    export_legs(legs)
    return player_contours


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


original_image = img_read(source)

height, width = original_image.shape

original_image_binary = binarize(original_image)
# skeleton = skeletonize(original_image_binary)
# inv_skeleton = invert(skeleton)

image = opening(original_image, 1)
image = closing(image, 1)
silhouette = binarize(image)

edges = Canny_edges(silhouette)
edges_array = edges_to_array(edges)
all_contours = copy.deepcopy(edges_array)

correct_contours = select_correct_contours(edges)

player_contours = draw_player_contours(correct_contours)
find_players(correct_contours, player_contours)

# plotting
plot_figures(invert(original_image_binary), invert(image), all_contours.astype(np.uint8),
             player_contours.astype(np.uint8))
