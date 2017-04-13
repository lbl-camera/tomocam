#! /usr/bin/env python

import gnufft
import afnumpy as afnp


vol = afnp.ones((25,256,256), dtype='f')
vol = vol + 1j * vol
fcn = afnp.zeros((25,256,256), dtype='complex64')
mrf_sigma = afnp.ones(1, dtype='f')
gnufft.add_hessian(mrf_sigma[0], vol, fcn)

print fcn[0,:10,:10]
print fcn[0,:10,:10]
