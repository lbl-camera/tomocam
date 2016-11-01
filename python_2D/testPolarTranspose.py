#! /usr/bin/env python

import gnufft
import numpy as np
import afnumpy as afnp
import arrayfire as af
from matplotlib import pyplot as plt

pts = afnp.ndarray((512, 128), dtype=np.complex64, af_array=af.randu(128, 512, dtype=af.Dtype.c32))
cplx = afnp.ndarray((512, 512), dtype=np.complex64, af_array=af.randu(512, 512, dtype=af.Dtype.c32))
kblut = afnp.ndarray((128,1), dtype=np.float32, af_array=af.randu(1, 128, dtype=af.Dtype.f32)) 
grid = [512, 512]
scale = 12
k_r = 5
for i in range(10000):
    res = gnufft.polarsample_transpose(pts, cplx, grid, kblut, scale, k_r)
print res.shape
print res.size
print res.dtype
print res[:5,:5]
plt.imshow(afnp.real(res));plt.show()
