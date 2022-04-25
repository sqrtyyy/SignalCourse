import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_isodata
from skimage.morphology import binary_opening


def remove_nested_contours(all_contours:list) -> list:
    to_remove = []
    all_contours = sorted([cv.approxPolyDP(x, 0.003 * cv.arcLength(x, True), True) for x in list(all_contours)],
                          key=cv.contourArea, reverse=True)
    for i in reversed(range(1, len(all_contours))):
        point = (int(all_contours[i][0][0][0]), int(all_contours[i][0][0][1]))
        for j in reversed(range(0, i)):
            if cv.pointPolygonTest(all_contours[j], point, False) == 1:
                to_remove.append(all_contours[i])
                break

    for cnt in to_remove:
        all_contours.remove(cnt)
    return  all_contours

class Finder:
    _contour: np.ndarray
    _white_list: np.ndarray
    _img: np.ndarray
    _img_without_list: np.ndarray
    _result_img: np.ndarray
    _items_contours: np.ndarray

    def __init__(self, img: np.ndarray):
        self._img = img.copy()
        self._contour = None
        self._white_list = None
        self._img_without_list = None
        self._result_img = None
        self._items_contours = None

    def get_white_contour(self) -> np.ndarray:
        if self._white_list is not None:
            return self._white_list

        hsv = cv.cvtColor(self._img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        _, threshed = cv.threshold(s, 60, 255, cv.THRESH_BINARY_INV)
        all_contours, _ = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        self._white_list = np.squeeze(max(all_contours, key=lambda item: cv.contourArea(item)))
        self._white_list = cv.approxPolyDP(self._white_list, 0.01 * cv.arcLength(self._white_list, True), True)

        if self._white_list.shape[0] != 4:
            raise "Wrong list detection"

        return self._white_list

    def find_contour(self) -> np.ndarray:
        hsv_min = np.array((0, 0, 0), np.uint8)
        hsv_max = np.array((10, 10, 10), np.uint8)

        if self._contour is not None:
            return self._contour
        hsv = cv.cvtColor(self._img, cv.COLOR_BGR2HSV)
        thresh = cv.inRange(hsv, hsv_min, hsv_max)
        all_contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        self._contour = np.squeeze(max(all_contours, key=lambda item: cv.contourArea(item)))
        self._contour = cv.approxPolyDP(self._contour, 0.01 * cv.arcLength(self._contour, True), True)

        return self._contour

    def __get_img_without_list(self) -> np.ndarray:
        if self._img_without_list is not None:
            return self._img_without_list
        background_color = (48, 100, 153)
        self._img_without_list = self._img.copy()

        cv.fillPoly(self._img_without_list, [self.get_white_contour()], color=background_color)
        return self._img_without_list

    def separate_items(self) -> np.ndarray:
        if self._items_contours is not None:
            return self._items_contours
        hsv_min = np.array((10, 100, 20), np.uint8)
        hsv_max = np.array((18, 255, 255), np.uint8)
        hsv = cv.cvtColor(self.__get_img_without_list(), cv.COLOR_BGR2HSV)

        thresh = cv.inRange(hsv, hsv_min, hsv_max)

        all_contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        min_contour_area = 20_000
        max_contour_area = 1_000_000
        areas_ratio = 0.6

        key = lambda x: max_contour_area > cv.contourArea(x) > min_contour_area and cv.contourArea(x) / cv.contourArea(
            cv.convexHull(x)) > areas_ratio

        all_contours = list(filter(key, all_contours))
        self._items_contours = np.asarray(remove_nested_contours(all_contours))
        return self._items_contours

    def save_result(self, name):
        if self._result_img is None:
            self.separate_items()
            self.get_white_contour()
            self.find_contour()

        self._result_img = self._img.copy()
        cv.drawContours(self._result_img, [self._contour], -1, (255, 0, 255), 6)
        cv.drawContours(self._result_img, [self._white_list], -1, (0, 255, 0), 6)
        cv.drawContours(self._result_img, self._items_contours, -1, (0, 0, 255), 6)

        cv.imwrite(f"{name}.jpg", self._result_img)
