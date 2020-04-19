from typing import List

import numpy as np

from src.data_processing.io.image import Image


def get_countour(image: Image, thickness: int = 1, color: int = 255) -> Image:
    matrix = np.zeros_like(image.img)
    for i in range(1, len(matrix) - 1):
        for j in range(1, len(matrix[i]) - 1):
            if (image.img[i][j] == 0) and (
                image.img[i + 1][j + 1] != 0
                or image.img[i + 1][j] != 0
                or image.img[i][j + 1] != 0
                or image.img[i - 1][j - 1] != 0
                or image.img[i - 1][j] != 0
                or image.img[i][j - 1] != 0
            ):
                matrix[i][j] = color
    return Image(matrix)


def histogram_segmentation(
    image: Image, target: float, sensitivity: int = 10, fidelity: int = 1, color: int = 255
) -> Image:
    return Image(
        _color_target(image.img, target, sensitivity, fidelity, color),
        title=f"{image.title}. Sens={sensitivity}. Prec={fidelity}.",
    )


def _color_target(arr: np.array, target: float, sensitivity: float, fidelity: int, color):
    if abs(np.mean(arr) - target) < sensitivity:
        return np.full_like(arr, color)
    elif max(arr.shape) <= fidelity:
        return np.zeros_like(arr)
    else:
        seg = _split_matrix(arr)
        return np.block(
            [
                [
                    _color_target(seg[0][0], target, sensitivity, fidelity, color),
                    _color_target(seg[0][1], target, sensitivity, fidelity, color),
                ],
                [
                    _color_target(seg[1][0], target, sensitivity, fidelity, color),
                    _color_target(seg[1][1], target, sensitivity, fidelity, color),
                ],
            ]
        )


def _split_matrix(arr):
    vcut, hcut = arr.shape[0] // 2, arr.shape[1] // 2
    return [np.hsplit(x, [hcut]) for x in np.vsplit(arr, [vcut])]


def _assemble_matrix(arr):
    return np.block([[arr[0][0], arr[0][1]], [arr[1][0], arr[1][1]]])
