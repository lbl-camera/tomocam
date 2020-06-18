#script to test tomopy

from __future__ import print_function, division
import matplotlib.pyplot as plt
import tomopy
import os
import dxchange as dx
from tomopy.prep.normalize import normalize

dataset=os.path.join(os.environ['HOME'], 'data', '20130807_234356_OIM121R_SAXS_5x.h5')
tomo, flats, darks, floc = dx.read_als_832h5(dataset,sino=(1000, 1003, 1))

print('Displaying  sinogram')
plt.imshow(tomo[:,0,:])
plt.show()

theta = tomopy.angles(tomo.shape[0])
tomo = normalize(tomo, flats, darks)
tomo = tomopy.remove_stripe_fw(tomo)
rec = tomopy.recon(tomo, theta, center=1294,algorithm='gridrec')
rec = tomopy.circ_mask(rec, 0)

for i in range(3):
    plt.imshow(rec[i])
    plt.show()
