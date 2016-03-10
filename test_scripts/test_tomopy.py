#script to test tomopy

from __future__ import print_function, division
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon
import tomopy
from normalize import normalize_bo

dataset='/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x.h5'
algorithm='sirt'
tomo, flats, darks, floc = tomopy.read_als_832h5(dataset,sino=(1000, 1003, 1))
print('Data read complete')

print('Displaying  sinogram')
imgplot = plt.imshow(tomo[:,0,:])

print('Generating angles')
theta = tomopy.angles(tomo.shape[0])

print('Normalization')
tomo = normalize_bo(tomo, flats, darks,20)

print('Ring removal')
tomo = tomopy.remove_stripe_fw(tomo)

print('Recon')
rec = tomopy.recon(tomo, theta, center=1294,algorithm='osem',emission=False)

print('Masking')
rec = tomopy.circ_mask(rec, 0)

tomopy.write_tiff_stack(rec, 'test.tiff')
