from scipy import signal

from src.base import number
from src.base.line import Line
from src.data_processing.io.image import Image


def bsf_pic(img: Image, low: number, high: number):
    result = Image(img.img.copy())
    rows, cols = result.img.shape
    for i in range(rows):
        row = result.img[i]
        result.img[i] = _bsf(row, low, high, len(row))
    return result


def bsf_line(line: Line, low: number, high: number):
    return Line(_bsf(line.y, low, high, fs=len(line.y)))


def _bsf(data, low, high, fs, order=5):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = signal.butter(order, [low, high], btype="bandstop")
    return signal.lfilter(b, a, data)
