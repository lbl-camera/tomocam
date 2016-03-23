#! /usr/bin/env python

import numpy as np
import afnumpy as afnp
from afnumpy import fft
import gnufft

def afnp_rand(n1, n2=1):
    arr = afnp.arrayfire.randu(n1, n2)
    return afnp.ndarray((n1, n2), dtype=np.float32, af_array=arr)


xi = afnp_rand(512, 180)
yi = afnp_rand(512, 180)
real  = afnp_rand(512, 180)
imag  = afnp_rand(512,180)
cplx = real + 1j * imag
grid = np.array([512, 512], dtype=np.int32)
kblut = afnp_rand(256)
scale = 12
k_r = 3
res = gnufft.polarsample(xi, yi, cplx, grid, kblut, scale, k_r)
print res
ones = afnp.arrayfire.constant(1, 512, 180)
res = fft.ifft2(res * ones)
