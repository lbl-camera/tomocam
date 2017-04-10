import arrayfire as af
import afnumpy as afnp
import numpy as np
import sys
#sys.path.append('/home/dkumar/tomocam')
#from tomocam.gnufft import tvd_update
from gnufft import tvd_update
import tomopy
import pyqtgraph as pg 

nslice = 50
im_size = 256
obj = tomopy.shepp3d((nslice,im_size,im_size))
x=obj[::2]
y=obj[1::2]
print(x.shape)
#x = np.ones((nslice,2000, 2000)).astype(np.float32)
#y = np.ones((nslice, 2000, 2000)).astype(np.float32)
vol = x + 1j * y
vol=255*afnp.array(vol.astype(np.complex64))
vol[0]=vol[0]*0
vol[-1]=vol[-1]*0

fcn = afnp.zeros((nslice/2, im_size, im_size), dtype=np.complex64)
tvd_update(vol, fcn)
print fcn.sum()
pg.image(np.real(np.array(vol)));pg.QtGui.QApplication.exec_()
pg.image(np.real(np.array(fcn)));pg.QtGui.QApplication.exec_()
