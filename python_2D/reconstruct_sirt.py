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
from XT_ForwardModel import forward_project, init_nufft_params, back_project

def main():

        oversamp_factor = 1.25 #For NUFFT
        parser = argparse.ArgumentParser()
        inputs = bl832inputs_parser(parser)

        num_slice = inputs['z_numElts']
        num_angles= inputs['num_views']/inputs['view_subsmpl_fact']
        af.set_device(inputs['gpu_device']) #Set the device number for gpu based code
        
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

        

        ################## GPU SIRT() ######################
        print('Starting GPU SIRT recon')

        new_tomo=np.transpose(tomo,(1,2,0))
        im_size =  new_tomo.shape[1]
        #Initialize structures for NUFFT
        sino={}
        geom={}
        sino['Ns'] = pad_size #Sinogram size after padding
        sino['Ns_orig'] = im_size #size of original sinogram
        sino['center'] = inputs['rot_center'] + (sino['Ns']/2 - sino['Ns_orig']/2)  #for padded sinogram
        sino['angles'] = theta

        #Initialize NUFFT parameters
        params = init_nufft_params(sino,geom)

        temp_y = afnp.zeros((sino['Ns'],num_angles),dtype=afnp.complex64)
        temp_x = afnp.zeros((sino['Ns'],sino['Ns']),dtype=afnp.complex64)
        x_recon  = afnp.zeros((num_slice/2,sino['Ns_orig'],sino['Ns_orig']),dtype=afnp.complex64) 
        pad_idx = slice(sino['Ns']/2-sino['Ns_orig']/2,sino['Ns']/2+sino['Ns_orig']/2)


        #Pre-compute diagonal scaling matrices ; one the same size as the image and the other the same as data
        #initialize an image of all ones
        x_ones= afnp.ones((sino['Ns_orig'],sino['Ns_orig']),dtype=afnp.complex64)
        temp_x[pad_idx,pad_idx]=x_ones
        temp_proj=forward_project(temp_x,params)*(sino['Ns']*afnp.pi/2)
        R = 1/afnp.abs(temp_proj)#(math.pi/2)*sino['Ns']*
        R[afnp.isnan(R)]=0
        R[afnp.isinf(R)]=0
        R=afnp.array(R,dtype=afnp.complex64)

        #Initialize a sinogram of all ones
        y_ones=afnp.ones((sino['Ns_orig'],num_angles),dtype=afnp.complex64)
        temp_y[pad_idx]=y_ones
        temp_backproj=back_project(temp_y,params)*nufft_scaling/2
        C = 1/(afnp.abs(temp_backproj))
        C[afnp.isnan(C)]=0
        C[afnp.isinf(C)]=0
        C=afnp.array(C,dtype=afnp.complex64)
        
        t=time.time()
        #Move all data to GPU
        slice_1=slice(0,num_slice,2)
        slice_2=slice(1,num_slice,2)
        gdata=afnp.array(new_tomo[slice_1]+1j*new_tomo[slice_2],dtype=afnp.complex64)
        
        #loop over all slices
        for i in range(0,num_slice/2):
          for iter_num in range(1,num_iter+1):
            #filtered back-projection
            temp_x[pad_idx,pad_idx]=x_recon[i]
            Ax = (math.pi/2)*sino['Ns']*forward_project(temp_x,params)
            temp_y[pad_idx]=gdata[i]
            x_recon[i] = x_recon[i]+(C*back_project(R*(temp_y-Ax),params)*nufft_scaling/2)[pad_idx,pad_idx]

        elapsed_time = (time.time()-t)
        print('Time for SIRT recon of %d slices : %f' % (num_slice,elapsed_time))

        #Move to CPU
        #Rescale result to match tomopy
        rec_sirt=np.array(x_recon,dtype=np.complex64)
        rec_sirt_final=np.zeros((num_slice,sino['Ns_orig'],sino['Ns_orig']),dtype=np.float32)
        rec_sirt_final[slice_1]=np.array(rec_sirt.real,dtype=np.float32)
        rec_sirt_final[slice_2]=np.array(rec_sirt.imag,dtype=np.float32)

        pg.image(rec_sirt_final);pg.QtGui.QApplication.exec_()

        print 'main: Done!'
main()


