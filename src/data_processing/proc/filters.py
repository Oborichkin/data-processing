import copy

import numpy as np
from scipy import signal

from src.base import number
from src.base.line import Line
from src.data_processing.io.image import Image


def lpf_line(line: Line, cut):
    return Line(_lpf(line.y, cut, fs=len(line.y)))


def bsf_line(line: Line, low: number, high: number):
    return Line(_bsf(line.y, low, high, fs=len(line.y)))


def lpf_pic(img: Image, cut):
    result = Image(img.img.copy())
    rows, cols = result.img.shape
    for i in range(rows):
        row = result.img[i]
        result.img[i] = _lpf(row, cut, len(row))
    return result


def bsf_pic(img: Image, low: number, high: number):
    result = Image(img.img.copy())
    rows, cols = result.img.shape
    for i in range(rows):
        row = result.img[i]
        result.img[i] = _bsf(row, low, high, len(row))
    return result


def _bsf(data, low, high, fs, order=5):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = signal.butter(order, [low, high], btype="bandstop")
    return signal.lfilter(b, a, data)


def _lpf(data, cut, fs, order=5):
    nyq = 0.5 * fs
    f_cut = cut / nyq
    b, a = signal.butter(order, f_cut, btype="low")
    return signal.lfilter(b, a, data)


def mean_filter(img: Image, size=5):
    matrix = _base_filter(img.img, size, lambda window: np.average(window))
    return Image(matrix)


def median_filter(img: Image, size=5):
    matrix = signal.medfilt2d(img.img, kernel_size=size)
    # matrix = __base_filter(img.matrix, window_size, lambda window: np.median(window))
    return Image(matrix)


def _base_filter(nparray, window_size, on_apply_window):
    radius = int(window_size / 2)
    matrix = nparray.copy()
    rows, cols = matrix.shape
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            window = []
            for i_window in range(i - radius, i + radius):
                for j_window in range(j - radius, j + radius):
                    window.append(nparray[i_window, j_window])
            matrix[i, j] = on_apply_window(window)
    return matrix
