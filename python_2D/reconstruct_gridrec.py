import tomopy
import argparse
import time
import os
import math
import numpy as np
import afnumpy as afnp
import arrayfire as af
import matplotlib.pyplot as plt
#import pyqtgraph as pg

from XT_argsParser import bl832inputs_parser
from normalize import normalize_bo
from XT_ForwardModel import forward_project, init_nufft_params, back_project

def main():
        
        parser = argparse.ArgumentParser()
        inputs = bl832inputs_parser(parser)

        num_slice = inputs['z_numElts']
        num_angles= inputs['num_views']/inputs['view_subsmpl_fact']
        af.set_device(inputs['gpu_device']) #Set the device number for gpu based code
        oversamp_factor = 1.25
        pad_size = np.int16(inputs['x_width']*oversamp_factor)
        fbp_filter_param=0.75
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

        fig = plt.figure()
        plt.imshow(tomo[:,1,:],cmap=plt.cm.Greys_r)
        fig.suptitle('Sinogram')
        pg.image(tomo)

        ################## GPU gridrec() ######################

        new_tomo=np.transpose(tomo,(1,2,0))
        im_size =  new_tomo.shape[1]
        #Initialize structures for NUFFT
        sino={}
        geom={}
        sino['Ns'] = pad_size #Sinogram size after padding
        sino['Ns_orig'] = im_size #size of original sinogram
        sino['center'] = inputs['rot_center'] + (sino['Ns']/2 - sino['Ns_orig']/2)  #for padded sinogram
        sino['angles'] = theta
        sino['filter'] = fbp_filter_param #Paramter to control strength of FBP filter normalized to [0,1]

        #Initialize NUFFT parameters
        params = init_nufft_params(sino,geom)

        rec_nufft = afnp.zeros((num_slice/2,sino['Ns_orig'],sino['Ns_orig']),dtype=afnp.complex64)
        Ax = afnp.zeros((sino['Ns'],num_angles),dtype=afnp.complex64)
        pad_idx = slice(sino['Ns']/2-sino['Ns_orig']/2,sino['Ns']/2+sino['Ns_orig']/2)

        t=time.time()
        #Move all data to GPU
        slice_1=slice(0,num_slice,2)
        slice_2=slice(1,num_slice,2)
        gdata=afnp.array(new_tomo[slice_1]+1j*new_tomo[slice_2],dtype=afnp.complex64)
        #loop over all slices
        for i in range(0,num_slice/2):
          Ax[pad_idx,:]=gdata[i]
          #filtered back-projection 
          rec_nufft[i] = (back_project(Ax,params))[pad_idx,pad_idx]

        elapsed_time = (time.time()-t)
        print('Time for NUFFT Back-proj of %d slices : %f' % (num_slice,elapsed_time))

        #Move to CPU
        #Rescale result to match tomopy
        rec_nufft=np.array(rec_nufft,dtype=np.complex64)*nufft_scaling

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



        
        fig = plt.figure()
        plt.imshow(np.abs(np.flipud(rec_tomopy[0])),cmap=plt.cm.Greys_r)
        plt.colorbar()
        fig.suptitle('Tomopy Gridrec Reconstruction')

        fig = plt.figure()
        plt.imshow(np.abs(rec_nufft[0].real),cmap=plt.cm.Greys_r)
        plt.colorbar()
        fig.suptitle('GPU NUFFT Reconstruction')

        print 'main: Done!'
        plt.show()
main()


