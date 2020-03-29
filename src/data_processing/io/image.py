import os
import random
import struct

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Image:
    def __init__(self, img: np.ndarray = None, title: str = None):
        self.img = img
        self.title = title

    @property
    def width(self):
        return self.img.shape[0]

    @property
    def height(self):
        return self.img.shape[1]

    def plot(self, dpi: int = 80, cmap="gray"):
        height, width = self.img.shape[0], self.img.shape[1]
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_title(self.title)
        plt.imshow(self.img, cmap=cmap)
        plt.show()

    def scale(self, ratio: float = 1, method: str = "nn") -> "Image":
        if method == "nn":
            return self._nearest_neighbour(ratio)
        elif method == "bilinear":
            return self._bilinear_interpolation(ratio)
        else:
            raise NotImplementedError

    def _nearest_neighbour(self, ratio) -> "Image":
        x, y, z = self.img.shape
        rescaled = np.empty((int(x * ratio), int(y * ratio), z), dtype=self.img.dtype)
        for x in range(rescaled.shape[0]):
            for y in range(rescaled.shape[1]):
                rescaled[x, y] = self.img[int(x / ratio), int(y / ratio)]
        return Image(img=rescaled)

    def _bilinear_interpolation(self, ratio) -> "Image":
        x, y, z = self.img.shape
        rescaled = np.empty((int(x * ratio), int(y * ratio), z), dtype=self.img.dtype)
        for i in range(rescaled.shape[0]):
            for j in range(rescaled.shape[1]):
                x, y = int(i / ratio), int(j / ratio)
                dx, dy = i / ratio - x, y / ratio - y
                x_safe = x if x + 1 == self.img.shape[0] else x + 1
                y_safe = y if y + 1 == self.img.shape[1] else y + 1
                A = self.img[x, y]
                B = self.img[x, y_safe]
                C = self.img[x_safe, y]
                D = self.img[x_safe, y_safe]
                rescaled[i, j] = A * (1 - dx) * (1 - dy) + B * (dx) * (1 - dy) + C * (dy) * (1 - dx) + D * (dx * dy)
        return Image(rescaled)

    def noise(self, intensity: float, mode="gauss") -> "Image":
        if mode == "gauss":
            return Image(
                self.img.copy()
                + np.random.normal(0, 255 * intensity, size=self.height * self.width).reshape(self.img.shape),
                title=f"{self.title} (Gaussian: {intensity})",
            )
        elif mode == "sp":
            result = self.img.copy()
            for px in np.nditer(result, op_flags=["readwrite"]):
                if random.random() < intensity:
                    sp = random.choice([0, 255])
                    px[...] = sp
            return Image(result, title=f"{self.title} (Salt&Pepper: {intensity})")

    @staticmethod
    def from_file(filepath: str, flags=cv2.IMREAD_GRAYSCALE) -> "Image":
        return Image(cv2.imread(filepath, flags=flags), title=os.path.basename(filepath))

    def from_xcr(filepath: str, w: int, h: int) -> "Image":
        with open(filepath, "rb") as f:
            result = list(struct.unpack(f"{w*h}h", f.read()))
        result = np.asarray(result).astype("float64")
        result = 255 * (result - result.min()) / result.ptp()
        result = result.reshape((w, h))
        return Image(result)
