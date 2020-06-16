import numpy as np
import os
import h5py
import tomocam


filename = '/data/tomochallange/phantom_00016/phantom_00016.h5'
h5fp = h5py.File(filename, 'r')

data = h5fp['/projs']
s = np.array(data[:,:16,:].transpose(1, 0, 2), copy=True)
h5fp.close()

nslc, nproj, nrow = s.shape
print(s.shape)
sino = tomocam.DistArray(s.copy())
model = tomocam.DistArray(np.ones((nslc, nrow, nrow), dtype=np.float32))
angs = np.linspace(0, np.pi, nproj, dtype=np.float32)

for i in range(20):
    grads = model.copy()
    print(' 000 ')
    tomocam.calc_gradients(grads, sino, angs, center=640)
    print(' 111 ')
    tomocam.update_total_variation(model, grads)
    print(' 222 ')
    tomocam.axpy(-0.1, grads, model)
    print(' 333 ')

