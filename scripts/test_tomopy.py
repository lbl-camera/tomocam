#script to test tomopy

from __future__ import print_function, division
import matplotlib.pyplot as plt
import tomopy
import os
import dxchange as dx
from tomopy.prep.normalize import normalize

DATADIR = '/home/dkumar/data'
FILENAME = 'tomo_00025/tomo_00025.h5'
dataset = os.path.join(DATADIR, FILENAME)
tomo, flat, dark, theta = dx.read_aps_32id(dataset,sino=(1000, 1016, 1))

tomo = normalize(tomo, flat, dark)
tomo = tomopy.remove_stripe_fw(tomo)
rec = tomopy.recon(tomo, theta, center=952,algorithm='gridrec')
rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

for i in range(3):
    plt.imshow(rec[i], origin='lower', cmap='gray')
    plt.show()
