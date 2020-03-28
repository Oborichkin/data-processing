import numpy as np

from src.base.line import Line
from src.data_processing.analysis import dft, idft


def convolution(a: Line, b: Line, mode="full"):
    return Line(np.convolve(a.y, b.y, mode=mode))


def deconvolution(a: Line, b: Line) -> Line:
    a_dft = dft(a)
    b_dft = dft(b)
    print(len(a_dft))
    print(len(b_dft))

    return idft(
        [
            complex(
                (a_dft[i].real * b_dft[i].real + a_dft[i].imag * b_dft[i].imag)
                / (b_dft[i].real ** 2 + b_dft[i].imag ** 2),
                (a_dft[i].imag * b_dft[i].real - a_dft[i].real * b_dft[i].imag)
                / (b_dft[i].real ** 2 + b_dft[i].imag ** 2),
            )
            for i in range(len(a_dft))
        ]
    )
