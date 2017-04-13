import arrayfire as af
import afnumpy as afnp
import numpy as np
import sys
from gnufft import tvd_update,add_hessian
import tomopy
import pyqtgraph as pg
import time
import arrayfire as af 

af.set_device(2)
nslice = 2
im_size = 256
#obj = np.ones((nslice,im_size,im_size),dtype=np.float32)
#obj=tomopy.shepp3d((nslice,im_size,im_size),dtype=np.float32)
obj =np.random.rand(nslice,im_size,im_size).astype(np.float32)
x=obj[::2]
y=obj[1::2]
print(x.shape)
vol = x + 1j * y
vol=afnp.array(vol.astype(np.complex64))#255*

fcn = afnp.zeros((nslice/2, im_size, im_size), dtype=np.complex64)
t=time.time()
tvd_update(1.2,1,vol, fcn)
elapsed = time.time()-t
print('Time taken for gradient %f' % (elapsed))
output = np.zeros((nslice,im_size,im_size),dtype=np.float32)
output[::2]=np.array(fcn).real
output[1::2]=np.array(fcn).imag
print(output.max())
print(output.min())
del fcn

fcn = afnp.zeros((nslice/2, im_size, im_size), dtype=np.complex64)
t=time.time()
add_hessian(2,vol, fcn)
elapsed = time.time()-t
print('Time taken for Hessian %f' % (elapsed))
output_hess = np.zeros((nslice,im_size,im_size),dtype=np.float32)
output_hess[::2]=np.array(fcn).real
output_hess[1::2]=np.array(fcn).imag
print(output_hess.max())
print(output_hess.min())
del fcn


#pg.image(obj);
#pg.image(output);
pg.image(output_hess);
pg.QtGui.QApplication.exec_()
