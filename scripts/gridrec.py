import os
import h5py
import numpy as np
import tomocam
import matplotlib.pyplot as plt

data_path = '/home/dkumar/Data/phantom'
filename = 'phantom_00016.h5'
data_file = os.path.join(data_path, filename)
if not os.path.isfile(data_file):
    print("error: file not found")
    exit(1)

fp = h5py.File(data_file, 'r')
data = fp['projs']
dims = [20, data.shape[0], data.shape[2]]
proj = np.zeros(dims, dtype=np.float32)
proj[:] = data[:,:20,:].transpose([1, 0, 2])
angs = np.linspace(0, np.pi, dims[1], dtype=np.float32)
sino = tomocam.DistArray(proj)
temp  = tomocam.iradon(sino, angs, center=640)
plt.imshow(temp[0])
plt.show()

