from typing import List

import numpy as np

from src.base.line import Line
from src.data_processing.io.image import Image


def histogram(img: Image):
    hist = np.zeros(256)
    for px in np.nditer(img.img, op_flags=["readwrite"]):
        hist[max(0, min(int(px), 255))] += 1
    return Line(hist)


def dft(line: Line) -> List[complex]:
    n = len(line)
    result = np.zeros(n, dtype=complex)
    for k in range(n):
        sumReal = 0
        sumImag = 0
        for t in range(n):
            angle = (2 * np.pi * k * t) / n
            sumReal += line.y[t] * np.cos(angle)
            sumImag += line.y[t] * np.sin(angle)
        result[k] = complex(sumReal / n, sumImag / n)
    return result


def get_frequency_spectrum(dft: List[complex]):
    return Line([abs(x) for x in dft][: len(dft) // 2])


def idft(vector: List[complex]) -> Line:
    n = len(vector)
    result = np.zeros(n)
    for k in range(n):
        sum_ = 0
        for t in range(n):
            angle = (2 * np.pi * k * t) / n
            sum_ += vector[t].real * np.cos(angle) + vector[t].imag * np.sin(angle)
        result[k] = sum_
    return Line(result)


def _idft(r, i):
    n = len(r)
    result = np.zeros(n)
    for k in range(n):
        sum_ = 0
        for t in range(n):
            angle = (2 * np.pi * k * t) / n
            sum_ += r[t] * np.cos(angle) + i[t] * np.sin(angle)
        result[k] = sum_
    return result


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def extract_range(img: Image, start: int, end: int, replace=True) -> Image:
    res = img.img.copy()
    for i in range(len(res)):
        for j in range(len(res[i])):
            if replace:
                res[i][j] = 255 if start < res[i][j] < end else 0
            else:
                res[i][j] = res[i][j] if start < res[i][j] < end else 0

    return Image(res)
