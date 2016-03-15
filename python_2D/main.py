import tomopy
import argparse
import time
import os
import math
import numpy as np
import afnumpy as afnp
import matplotlib.pyplot as plt
import pyqtgraph as pg
import afnumpy as afnp
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

from XT_argsParser import bl832inputs_parser
#from XT_ForwardModel import forward_project
from normalize import normalize_bo

def main():
        
        parser = argparse.ArgumentParser()
        inputs = bl832inputs_parser(parser)

        algorithm='sirt'
        tomo, flats, darks, floc = tomopy.read_als_832h5(inputs['input_hdf5'],ind_tomo=range(1,inputs['num_views'],inputs['view_subsmpl_fact']),sino=(inputs['z_start'], inputs['z_start']+inputs['z_numElts'], 1))
        print('Data read complete')
        print tomo.shape

        print('Displaying  sinogram')
        #imgplot = plt.imshow(tomo[:,0,:])

        print('Generating angles')
        theta = tomopy.angles(tomo.shape[0])
        print theta.shape

        #Need to modify to return the raw counts for noise estimation 
        print('Normalization')
        tomo,weight = normalize_bo(tomo, flats, darks,inputs['num_dark'])

        print('Ring removal')
        tomo = tomopy.remove_stripe_fw(tomo)

        fig = plt.figure()
#        plt.imshow(tomo[:,1,:],cmap=plt.cm.Greys_r)
#        fig.suptitle('Sinogram')
        pg.image(tomo)

        #Change this to a mbir.recon 
        print('Recon')
        rec = tomopy.recon(tomo, theta, center=inputs['rot_center'],algorithm=algorithm,emission=False)
        # gnufft.polarsample()

        print('Masking')
        rec = tomopy.circ_mask(rec, 0)
        print rec.shape

        tomopy.write_tiff_stack(rec, 'test.tiff')

        fig = plt.figure()
#        plt.imshow(rec[1,:,:],cmap=plt.cm.Greys_r)
#        fig.suptitle('Reconstruction')

        pg.image(rec)
        
        temp1 = afnp.array(tomo)
        temp2 = afnp.array(rec)
        temp3 = afnp.array(weight)

        print 'main: Done!'
        plt.show()
main()


