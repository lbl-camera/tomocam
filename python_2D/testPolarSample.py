#! /usr/bin/env python

import gnufft
import numpy as np
import afnumpy as afnp
import arrayfire as af

def afnp_rand(n1, n2=1):
    arr = afnp.arrayfire.randu(n1, n2)
    return afnp.ndarray((n1, n2), dtype=np.float32, af_array=arr)


pts = afnp.ndarray((512, 128), dtype=np.complex, af_array=af.randu(128, 512, dtype=af.Dtype.c32))
cplx = afnp.ndarray((512, 512), dtype=np.complex, af_array=af.randu(512, 512, dtype=af.Dtype.c32))
kblut = afnp.ndarray((128,1), dtype=np.float, af_array=af.randu(1, 128, dtype=af.Dtype.f32)) 
#print kblut[:10]
scale = 12
k_r = 3
res = gnufft.polarsample(pts, cplx, kblut, scale, k_r)
print res
