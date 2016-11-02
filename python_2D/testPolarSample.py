#! /usr/bin/env python

import gnufft
import numpy as np
import afnumpy as afnp
import arrayfire as af
import time

def afnp_rand(n1, n2=1):
    arr = afnp.arrayfire.randu(n1, n2)
    return afnp.ndarray((n1, n2), dtype=np.float32, af_array=arr)


pts = afnp.ndarray((512, 128), dtype=np.complex64, af_array=af.randu(128, 512, dtype=af.Dtype.c32))
cplx = afnp.ndarray((512, 512), dtype=np.complex64, af_array=af.randu(512, 512, dtype=af.Dtype.c32))
kblut = afnp.ndarray((128,1), dtype=np.float32, af_array=af.randu(1, 128, dtype=af.Dtype.f32)) 
#print kblut[:10]
scale = 12
k_r = 3
t0 = time.time()
res = gnufft.polarsample(pts, cplx, kblut, scale, k_r)
dt = time.time() - t0
print "time: "+str(dt)
print res.shape
print res.size
print res.dtype
print res
n = res.shape[0]
r = np.random.rand(n,1).astype(np.float32)
row = afnp.ndarray((n,1), dtype=np.float32, buffer=r)
print row.shape
res * row
