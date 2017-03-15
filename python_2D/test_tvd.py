import arrayfire as af
import afnumpy as np
import sys
sys.path.append('/home/dkumar/tomocam')
from tomocam.gnufft import tvd_update

x = np.random.rand(3, 4, 5).astype(np.float32)
y = np.random.rand(3, 4, 5).astype(np.float32)
vol = x + 1j * y
fcn = np.zeros((3, 4, 5), dtype=np.complex64)
tvd_update(vol, fcn)
print fcn

