import numpy as np
import os
import h5py
import tomocam
import matplotlib.pyplot as plt


filename = '/home/dkumar/data/phantom_00017/phantom_00017.h5'
h5fp = h5py.File(filename, 'r')
data = np.copy(h5fp['/projs'][:,:128,:])
angs = np.copy(h5fp['/angs'])
h5fp.close()

print(data.shape)
sino = np.transpose(data, [1, 0, 2])

recon = tomocam.MBIR(sino, angs, 640, 10, over_sample=2, smoothness=1)

plt.imshow(recon[0])
plt.show()
