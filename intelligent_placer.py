# %%
import time
from typing import Tuple, Any

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

import img_processing
import numpy as np
import cv2 as cv


def get_contour_boundaries(contour: np.ndarray) -> Tuple[int, int, int, int]:
    extLeft = tuple(contour[contour[:, :, 0].argmin()][0])[0]
    extRight = tuple(contour[contour[:, :, 0].argmax()][0])[0]
    extTop = tuple(contour[contour[:, :, 1].argmin()][0])[1]
    extBot = tuple(contour[contour[:, :, 1].argmax()][0])[1]
    return extLeft, extRight, extTop, extBot


def shift_contour(contour: np.ndarray):
    extLeft, _, extTop, _ = get_contour_boundaries(contour)
    return np.subtract(contour, [extLeft, extTop])


def get_countour_area(contour: np.ndarray) -> np.ndarray:
    shifted_contour = shift_contour(contour)
    extLeft, extRight, extTop, extBot = get_contour_boundaries(shifted_contour)
    contour_area = np.zeros((extBot, extRight))
    cv.fillPoly(contour_area, pts=[shifted_contour], color=(255, 255, 255))
    return contour_area


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def try_to_place(contour: np.ndarray, items_contours: list) -> bool:
    contour_area = cv.resize(get_countour_area(contour), (0, 0), fx=0.2, fy=0.2)
    angle_step = 15
    cur_item_idx = 0
    rotates_stack = [0]
    area_stack = [contour_area.copy()]
    rotates_limits = np.zeros(len(items_contours))
    rotates_limits.fill(360)

    while cur_item_idx != -1 and cur_item_idx != len(items_contours):

        if rotates_stack[-1] >= rotates_limits[cur_item_idx]:
            cur_item_idx -= 1
            area_stack.pop()
            rotates_stack.pop()
            if len(rotates_stack) > 0:
                rotates_stack[-1] += angle_step
            continue
        item = cv.resize(get_countour_area(rotate_contour(items_contours[cur_item_idx], rotates_stack[-1])), (0, 0),
                         fx=0.2, fy=0.2)
        _, item = cv.threshold(item, 1, 255, cv.THRESH_BINARY)

        if rotates_limits[cur_item_idx] == 360:
            reversed_item = cv.flip(item, 1)
            if np.sum(cv.bitwise_and(item, reversed_item)) / np.sum(cv.bitwise_or(item, reversed_item)) > 0.9:
                rotates_limits[cur_item_idx] = 180

        cur_contour_area = area_stack[-1].copy()
        item_width = item.shape[1]
        item_height = item.shape[0]
        suitable_height = None
        suitable_width = None
        for width in range(0, cur_contour_area.shape[1] - item_width, 10):
            for height in range(0, cur_contour_area.shape[0] - item_height, 10):
                cur_area = cur_contour_area[height:item_height + height, width:item_width + width]
                if np.array_equal(cv.bitwise_and(item, cur_area), item):
                    suitable_height = height

            if suitable_height:
                suitable_width = width
                break
        if suitable_height is None or suitable_width is None:
            rotates_stack[-1] += angle_step
            continue
        for x in range(item_width):
            for y in range(item_height):
                cur_contour_area[y + suitable_height, x + suitable_width] = 0 if item[y, x] == 255 else \
                cur_contour_area[y + suitable_height, x + suitable_width]
        area_stack.append(cur_contour_area.copy())

        rotates_stack.append(0)
        cur_item_idx += 1

    if cur_item_idx == len(items_contours):
        plt.imshow(area_stack[-1])
        plt.show()
    return cur_item_idx == len(items_contours)


def check_image(path_to_img: str):
    img = cv.imread(path_to_img)
    plt.imshow(img)
    plt.show()
    finder = img_processing.Finder(img)
    contour = finder.find_contour()
    items_contours = finder.separate_items()
    if sum([cv.contourArea(cnt) for cnt in items_contours]) > cv.contourArea(contour):
        return False
    return try_to_place(contour, items_contours)
