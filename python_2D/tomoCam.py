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

from XT_ForwardModel import forward_project, init_nufft_params, back_project

#Gridrec reconstruction using GPU based gridding
#Inputs: tomo : 3D numpy sinogram array with dimensions same as tomopy
#        angles : Array of angles in radians
#        center : Floating point center of rotation
#       input_params : A dictionary with the keys
#        'gpu_device' : Device id of the gpu (For a 4 GPU cluster ; 0-3)
#       'oversamp_factor': A factor by which to pad the image/data for FFT
#       'fbp_filter_param' : A number between 0-1 for setting the filter cut-off for FBP

def gpuGridrec(tomo,angles,center,input_params):        
        print('Starting GPU NUFFT recon')
        #allocate space for final answer 
        af.set_device(input_params['gpu_device']) #Set the device number for gpu based code
        #Change tomopy format
        new_tomo=np.transpose(tomo,(1,2,0)) #slice, columns, angles
        im_size =  new_tomo.shape[1]
        num_slice = new_tomo.shape[0]
        num_angles=new_tomo.shape[2]
        pad_size=np.int16(im_size*input_params['oversamp_factor'])
        nufft_scaling = (np.pi/pad_size)**2
        #Initialize structures for NUFFT
        sino={}
        geom={}
        sino['Ns'] =  pad_size#Sinogram size after padding
        sino['Ns_orig'] = im_size #size of original sinogram
        sino['center'] = center + (sino['Ns']/2 - sino['Ns_orig']/2)  #for padded sinogram
        sino['angles'] = angles
        sino['filter'] = input_params['fbp_filter_param'] #Paramter to control strength of FBP filter normalized to [0,1]

        #Initialize NUFFT parameters
        nufft_params = init_nufft_params(sino,geom)
        rec_nufft = afnp.zeros((num_slice/2,sino['Ns_orig'],sino['Ns_orig']),dtype=afnp.complex64)
        Ax = afnp.zeros((sino['Ns'],num_angles),dtype=afnp.complex64)
        pad_idx = slice(sino['Ns']/2-sino['Ns_orig']/2,sino['Ns']/2+sino['Ns_orig']/2)
        rec_nufft_final=np.zeros((num_slice,sino['Ns_orig'],sino['Ns_orig']),dtype=np.float32)
        
        t=time.time()
        #Move all data to GPU
        slice_1=slice(0,num_slice,2)
        slice_2=slice(1,num_slice,2)
        gdata=afnp.array(new_tomo[slice_1]+1j*new_tomo[slice_2],dtype=afnp.complex64)
        x_recon = afnp.zeros((sino['Ns'],sino['Ns']),dtype=afnp.complex64)
        #loop over all slices
        for i in range(0,num_slice/2):
          Ax[pad_idx,:]=gdata[i]
          #filtered back-projection 
          rec_nufft[i] = (back_project(Ax,nufft_params))[pad_idx,pad_idx]

        elapsed_time = (time.time()-t)
        print('Time for NUFFT Back-proj of %d slices : %f' % (num_slice,elapsed_time))

        #Move to CPU
        #Rescale result to match tomopy
        rec_nufft=np.array(rec_nufft,dtype=np.complex64)*nufft_scaling
        rec_nufft_final[slice_1]=np.array(rec_nufft.real,dtype=np.float32)
        rec_nufft_final[slice_2]=np.array(rec_nufft.imag,dtype=np.float32)
        return rec_nufft_final
