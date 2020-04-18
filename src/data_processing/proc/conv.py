import numpy as np

from src.base.line import Line
from src.data_processing.analysis import _idft, dft, idft
from src.data_processing.io.image import Image


def convolution(a: Line, b: Line, mode="full"):
    return Line(np.convolve(a.y, b.y, mode=mode))


def deconvolution(a: Line, b: Line) -> Line:
    a_dft = dft(a)
    b_dft = dft(b)

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


def deconv_pic(img, kernel):
    result = Image(img.img.copy())
    rows, cols = result.img.shape
    for i in range(rows):
        row = result.img[i]
        result.img[i] = _deconvolution(Line(row), kernel)
    return result


def deconv_pic_reg(img, kernel, reg):
    result = Image(img.img.copy())
    rows, cols = result.img.shape
    for i in range(rows):
        row = result.img[i]
        result.img[i] = _regularized_deconv(Line(row), kernel, reg)
    return result


def _complex_division(ar, ai, br, bi):
    ar = np.array(ar)
    ai = np.array(ai)
    br = np.array(br)
    bi = np.array(bi)
    divider = br ** 2 + bi ** 2
    r = (ar * br + ai * bi) / divider
    i = (ai * br - ar * bi) / divider
    return r, i


def _deconvolution(a, b):
    a_dft = dft(a)
    ar, ai = [x.real for x in a_dft], [x.imag for x in a_dft]
    b_dft = dft(b)
    br, bi = [x.real for x in b_dft], [x.imag for x in b_dft]
    cr, ci = _complex_division(ar, ai, br, bi)
    return _idft(cr, ci)


def _complex_product(r1, i1, r2, i2):
    r = r1 * r2 - i1 * i2
    i = r1 * i2 + r2 * i1
    return r, i


def _regularized_deconv(a, b, k):
    a_dft = dft(a)
    ar, ai = [x.real for x in a_dft], [x.imag for x in a_dft]
    b_dft = dft(b)
    br, bi = [x.real for x in b_dft], [x.imag for x in b_dft]
    ar = np.array(ar)
    ai = np.array(ai)
    br = np.array(br)
    bi = np.array(bi)
    square = br ** 2 + bi ** 2 + k
    cr, ci = br / square, -bi / square
    dr, di = _complex_product(cr, ci, ar, ai)
    return _idft(dr, di)
