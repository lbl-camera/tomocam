#! /usr/bin/env python

import arrayfire as af
import cmath

def fftshift(arr, center = None):
    if center is None:
        if arr.numdims() == 1:
            if arr.is_real():
                return arr * (-1)**af.range(arr.shape[0])
        elif arr.numdims() == 2:
            x = (-1)**af.range(arr.shape[0])
            y = (-1)**af.range(arr.shape[1])
            return arr * af.matmul(x, y.T)
        else:
            raise TypeError('Unsupportd array shape')
    else:
        N = arr.shape[0]
        t = 1j * 2. * cmath.pi * center / N
        return arr * (cmath.exp(t) * af.range(N))

def ifftshift(arr, center):
    N = arr.shape[0]
    t = -1j * 2 * cmath.pi * center / N
    return arr * (cmath.exp(t) * af.range(N)) 
