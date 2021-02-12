import numpy as np
import os
import h5py
import tomocam
import matplotlib.pyplot as plt


filename = '/home/dkumar/Data/phantom/phantom_00016.h5'
h5fp = h5py.File(filename, 'r')

data = h5fp['/projs']
s = np.array(data[:,:16,:].transpose(1, 0, 2), copy=True)
h5fp.close()

nslc, nproj, nrow = s.shape
print(s.shape)
sino = tomocam.DistArray(s.copy())
model = tomocam.DistArray(np.ones((nslc, nrow, nrow), dtype=np.float32))
angs = np.linspace(0, np.pi, nproj, dtype=np.float32)

lam = 0.8
prev_e = np.Infinity

for i in range(20):
    grads = model.copy()
    tomocam.calc_gradients(grads, sino, angs, center=640)
    tomocam.update_total_variation(model, grads)
    e = grads.norm() * lam
    if e > prev_e:
        lam *= 0.9
    else: 
        tomocam.axpy(-lam, grads, model)
        prev_e = e
        print(e)
recon = model.to_numpy()

for i in range(10):
    plt.imshow(recon[i])
    plt.show()
