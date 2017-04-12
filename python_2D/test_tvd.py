import arrayfire as af
import afnumpy as afnp
import numpy as np
import sys
from gnufft import tvd_update
import tomopy
import pyqtgraph as pg
import time
import arrayfire as af 

af.set_device(2)
nslice = 256
im_size = 500
obj = np.ones((nslice,im_size,im_size),dtype=np.float32)
#obj=tomopy.shepp3d((nslice,im_size,im_size))
#obj =np.random.rand(nslice,im_size,im_size).astype(np.float32)
x=obj[::2]
y=obj[1::2]
print(x.shape)
vol = x + 1j * y
vol=afnp.array(vol.astype(np.complex64))#255*

fcn = afnp.zeros((nslice/2, im_size, im_size), dtype=np.complex64)
t=time.time()
tvd_update(vol, fcn)
elapsed = time.time()-t
print('Time taken %f' % (elapsed))
output = np.zeros((nslice,im_size,im_size),dtype=np.float32)
output[::2]=np.array(fcn).real
output[1::2]=np.array(fcn).imag
print(output.max())
print(output.min())
pg.image(obj);pg.QtGui.QApplication.exec_()
pg.image(output);pg.QtGui.QApplication.exec_()
