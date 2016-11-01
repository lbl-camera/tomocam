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
from tomoCam import gpuGridrec
#from XT_ForwardModel import forward_project, init_nufft_params, back_project


def main():
        
        parser = argparse.ArgumentParser()
        inputs = bl832inputs_parser(parser)

        num_slice = inputs['z_numElts']
        num_angles= inputs['num_views']/inputs['view_subsmpl_fact']
        
        oversamp_factor = 1.25
        pad_size = np.int16(inputs['x_width']*oversamp_factor)
        fbp_filter_param=inputs['fbp_filter_param']
        nufft_scaling = (np.pi/pad_size)**2
        
        algorithm='gridrec'
        
        tomo, flats, darks, floc = tomopy.read_als_832h5(inputs['input_hdf5'],ind_tomo=range(1,inputs['num_views']+1,inputs['view_subsmpl_fact']),sino=(inputs['z_start'], inputs['z_start']+inputs['z_numElts'], 1))
        print('Data read complete')


        print('Displaying  sinogram')
        #imgplot = plt.imshow(tomo[:,0,:])

        print('Generating angles')
        theta = tomopy.angles(num_angles)

        #Need to modify to return the raw counts for noise estimation 
        print('Normalization')
        tomo,weight = normalize_bo(tomo, flats, darks,inputs['num_dark'])

        print('Ring removal')
        tomo = tomopy.remove_stripe_fw(tomo)

#        fig = plt.figure()
#        plt.imshow(tomo[:,1,:],cmap=plt.cm.Greys_r)
#        fig.suptitle('Sinogram')
        

        ################## GPU gridrec() ######################
        input_params={}
        input_params['gpu_device']=inputs['gpu_device']
        input_params['fbp_filter_param']=inputs['fbp_filter_param']
        input_params['oversamp_factor']=oversamp_factor

        t=time.time()
        rec_nufft_final=gpuGridrec(tomo,theta,inputs['rot_center'],input_params)
        elapsed_time=time.time()-t
        print('Time for NUFFT Gridrec of %d slices : %f' % (num_slice,elapsed_time))

        ##################TomoPy Recon#####################
        print('Recon - tomopy GridRec')
        t=time.time()
        rec_tomopy = tomopy.recon(tomo, theta, center=inputs['rot_center'],algorithm=algorithm)#emission=False)
        elapsed_time = (time.time()-t)
        print('Time for reconstucting using Tomopy GridRec of %d slices : %f' % (num_slice,elapsed_time))

#       print('Recon - tomopy Astra')
#       t=time.time()
#       options = {'proj_type':'cuda', 'method':'FBP_CUDA'}
#       rec_astra = tomopy.recon(tomo, theta, center=inputs['rot_center'], algorithm=tomopy.astra, options=options)
#       elapsed_time = (time.time()-t)
#       print('Time for reconstucting using Tomopy Astra of %d slices : %f' % (num_slice,elapsed_time))

        
#        fig = plt.figure()
#        plt.imshow(np.abs(np.flipud(rec_tomopy[0])),cmap=plt.cm.Greys_r)
#        plt.colorbar()
#        fig.suptitle('Tomopy Gridrec Reconstruction')

#        fig = plt.figure()
#        plt.imshow(rec_nufft[0].real,cmap=plt.cm.Greys_r)
#        plt.colorbar()
#        fig.suptitle('GPU NUFFT Reconstruction')

        pg.image(rec_nufft_final);pg.QtGui.QApplication.exec_()

        print 'main: Done!'
#        plt.show()
main()


