#Script to test afnumpy

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import afnumpy as afnp
import afnumpy.fft as fft
import time 

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

length = 10^7

t = time.time()
temp = afnp.array(afnp.random.random((1,length)))
temp2 = afnp.array(afnp.random.random((1,length)))
temp3 = temp+temp2
elapsed1 = time.time() - t
print(elapsed1)

t = time.time()
temp = np.array(np.random.random((1,length)))
temp2 = np.array(np.random.random((1,length)))
temp3 = temp+temp2
elapsed2 = time.time() - t
print(elapsed2)
