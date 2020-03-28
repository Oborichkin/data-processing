import numpy as np

from src.base import number, vector


class Line:
    def __init__(self, y):
        self.y = y

    def __len__(self):
        return len(self.y)

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
