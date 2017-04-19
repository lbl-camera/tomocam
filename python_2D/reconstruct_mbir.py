import tomopy
import argparse
import time
import os
import math
import numpy as np
import afnumpy as afnp
import arrayfire as af
import matplotlib.pyplot as plt
import pyqtgraph as pg

from XT_argsParser import bl832inputs_parser
from normalize import normalize_bo
from tomoCam import gpuMBIR


def main():

        oversamp_factor = 1.25 #For NUFFT
        parser = argparse.ArgumentParser()
        inputs = bl832inputs_parser(parser)

        num_slice = inputs['z_numElts']
        num_angles= inputs['num_views']/inputs['view_subsmpl_fact']
        
        pad_size = np.int16(inputs['x_width']*oversamp_factor)
        num_iter = inputs['num_iter']
        nufft_scaling = (np.pi/pad_size)**2
        
        
        tomo, flats, darks, floc = tomopy.read_als_832h5(inputs['input_hdf5'],ind_tomo=range(1,inputs['num_views']+1,inputs['view_subsmpl_fact']),sino=(inputs['z_start'], inputs['z_start']+inputs['z_numElts'], 1))
        print('Data read complete')


        print('Generating angles')
        theta = tomopy.angles(num_angles)

        #Need to modify to return the raw counts for noise estimation 
        print('Normalization')
        tomo,weight = normalize_bo(tomo, flats, darks,inputs['num_dark'])

        print('Ring removal')
        tomo = tomopy.remove_stripe_fw(tomo)
        
        ################## GPU MBIR ######################
        input_params={}
        input_params['gpu_device']=inputs['gpu_device']
        input_params['oversamp_factor']=oversamp_factor
        input_params['num_iter']=num_iter
        input_params['p']=inputs['p']
        input_params['smoothness']=inputs['smoothness']
        t=time.time()
        rec_mbir_final = gpuMBIR(tomo,theta,inputs['rot_center'],input_params)
        elapsed_time = (time.time()-t)
        print('Time for reconstucting using GPU-MBIR of %d slices with %d iter : %f' % (num_slice,num_iter,elapsed_time))

        pg.image(rec_mbir_final);pg.QtGui.QApplication.exec_()
#        np.save('/home/svvenkatakrishnan/results/mbir_notch1080_70slice',rec_mbir_final)
        
#        fig = plt.figure()
#        sirt_camTomo = rec_sirt_final[124]
#        plt.imshow(sirt_camTomo,cmap=plt.cm.Greys_r,vmin=0,vmax=0.00075)
#        plt.colorbar()
#        fig.suptitle('Tomopy GPU-SIRT Reconstruction')
#        plt.draw()
#        plt.show()

        print 'main: Done!'
main()


