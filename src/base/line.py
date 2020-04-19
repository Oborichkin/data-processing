import struct

import matplotlib.pyplot as plt
import numpy as np

from src.base import number, vector


class Line:
    def __init__(self, y, x=None):
        assert x is None or len(y) == len(x)
        self.y = y
        self.x = x if x is not None else range(len(y))

    def __len__(self):
        return len(self.y)

    def plot(self):
        plt.figure(figsize=(20, 5))
        plt.plot(self.x, self.y)

    def __pow__(self: "Line", other: "Line") -> "Line":
        y = list()
        y.extend(self.y)
        y.extend(other.y)
        return Line(y=y)

    @staticmethod
    def from_dat(filepath: str) -> "Line":
        with open(filepath, "rb") as f:
            result = f.read()
        return Line(y=struct.unpack(f"{len(result)//4}f", result))

    @staticmethod
    def linear(a: number = 1, b: number = 0, range_: vector = np.arange(0, 1, 0.1)) -> "Line":
        return Line(y=[a * x + b for x in range_])

    @staticmethod
    def exponential(alpha: number = 1, beta: number = 1, range_: vector = np.arange(0, 1, 0.1)) -> "Line":
        return Line(y=[beta * np.e ** (alpha * x) for x in range_])

    @staticmethod
    def harmonic(
        freq: number = 1, amp: number = 1, phase: number = 0, range_: vector = np.arange(0, 2 * np.pi, 0.1)
    ) -> "Line":
        return Line([np.sin(2 * np.pi * x * freq + phase) * amp for x in range_])
