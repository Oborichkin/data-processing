import numpy as np

from src.base.line import Line
from src.data_processing.proc import convolution


class Cardiogram(Line):
    def __init__(self, dots, freq, dt, interval, relaxation):
        self.base = self._base(dots, freq, dt, relaxation)
        self.delta = self._delta(dots, interval)
        self.y = convolution(self.base, self.delta).y[: len(self.base)]

    def _base(self, dots, freq, dt, relaxation):
        multiplier = 2 * np.pi * freq * dt
        return Line([np.sin(x * multiplier) * np.exp(-relaxation * dt * x) for x in range(dots)])

    def _delta(self, dots, interval):
        return Line([1 if x % interval == 0 else 0 for x in range(dots)])
