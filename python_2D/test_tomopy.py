#script to test tomopy

from __future__ import print_function, division
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon
import tomopy
from normalize import normalize_bo

dataset='/home/svvenkatakrishnan/data/20131106_074854_S45L3_notch_OP_10x.h5'
#20130807_234356_OIM121R_SAXS_5x.h5'
algo='fbp'
num_sets = 2

tomo, flats, darks, floc = tomopy.read_als_832h5(dataset,sino=(1000, 1010, 1))
print('Data read complete')

print('Displaying  sinogram')
imgplot = plt.imshow(tomo[:,0,:]);plt.show();

print('Generating angles')
theta = tomopy.angles(tomo.shape[0])

print('Normalization')
tomo,weight = normalize_bo(tomo, flats, darks,num_sets)

#print('Ring removal')
#tomo = tomopy.remove_stripe_fw(tomo)

print('Recon')
rec = tomopy.recon(tomo, theta, center=1328,algorithm=algo,emission=False)#1294

print('Masking')
rec = tomopy.circ_mask(rec, 0)

imgplot = plt.imshow(rec[:,0,:]);plt.show();

tomopy.write_tiff_stack(rec, 'test.tiff')
