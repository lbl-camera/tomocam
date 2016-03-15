#! /usr/bin/env python

import numpy as np
import afnumpy as afnp
import gnufft

xi = afnp.random.rand(512,180).astype(np.float32)
yi = afnp.random.rand(512,180).astype(np.float32)
real  = afnp.random.rand(512,512).astype(np.float32)
imag  = afnp.random.rand(512,512).astype(np.float32)
x = real + 1j * imag
grid = afnp.array([512, 512], dtype=np.int32)
kblut = afnp.random.rand(256).astype(np.float32)
scale = 12
k_r = 3
res = gnufft.polarsample(xi, yi, x, grid, kblut, scale, k_r)
