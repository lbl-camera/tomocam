#! /usr/bin/env python

import arrayfire as af
import cmath
from .utils import multiply

def fftshift(arr, center = None):
    if center is None:
        if arr.numdims() == 1:
            if arr.is_real():
                return multiply(arr, (-1)**af.range(arr.shape[0]))
        elif arr.numdims() == 2:
            x = (-1)**af.range(arr.shape[0])
            y = (-1)**af.range(arr.shape[1])
            return multiply(arr, af.matmul(x, y.T))
        else:
            raise TypeError('Unsupportd array shape')
    else:
        N = arr.shape[0]
        t = 1j * 2. * cmath.pi * center / N
        return multiply(arr, (cmath.exp(t) * af.range(N)))

def ifftshift(arr, center):
    N = arr.shape[0]
    t = -1j * 2 * cmath.pi * center / N
    return arr * (cmath.exp(t) * af.range(N)) 
