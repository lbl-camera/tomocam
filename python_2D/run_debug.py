#! /usr/bin/env python

import numpy as np
import afnumpy as afnp
import afnumpy.fft as fft
import arrayfire as af
import gnufft

a = af.randu(512)
real  = afnp.ndarray((512,1), dtype=np.float32, af_array = a)
a = af.randu(512)
imag  = afnp.ndarray((512,1), dtype=np.float32, af_array = a)
cplx = real + 1j * imag
a = af.constant(1, 512, dtype=af.Dtype.f32)
ones = afnp.ndarray((512,1), dtype=np.float32, af_array = a)
res = gnufft.debug(real, imag)
t1 = res * ones
t2 = fft.ifft(t1)
