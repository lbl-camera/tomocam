import tomopy
import argparse
import time
import os
import numpy as np
import afnumpy as afnp
import matplotlib.pyplot as plt
#import pyqtgraph as pg
import afnumpy as afnp
import math
import scipy.io as scio
import h5py

from XT_argsParser import bl832inputs_parser
from XT_ForwardModel import forward_project, init_nufft_params, back_project
from XT_Common import padmat,padmat_v2
from normalize import normalize_bo

fpath='/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x_Full.mat'
f = h5py.File(fpath)
norm_data=f['norm_data']
norm_data=np.transpose(norm_data,(2,1,0))
print norm_data.shape

num_slice =  norm_data.shape[0]
num_angles= norm_data.shape[2]
im_size =  norm_data.shape[1] #A n X n X n volume
sino_center = 1294
slice_idx = num_slice/2
ang = tomopy.angles(num_angles+1) # Generate uniformly spaced tilt angles.
ang=ang[:-1]

sino={}
geom={}
sino['Ns'] = 3200#im_size*2 #Sinogram size after padding
sino['Ns_orig'] = im_size #size of original sinogram
sino['center'] = sino_center + (sino['Ns']/2 - sino['Ns_orig']/2)  #for padded sinogram
sino['angles'] = ang
sino['filter'] = 1 #Paramter to control strength of FBP filter normalized to [0,1]

params = init_nufft_params(sino,geom)

#norm_data=afnp.array(norm_data[:40],dtype=afnp.complex64)
#temp_mat=afnp.array(np.zeros((sino['Ns'],num_angles)),dtype=afnp.complex64)

t=time.time()
#loop over all slices
for i in range(500,551):
  #pad data array and move it to GPU 
  Ax = afnp.array(padmat(norm_data[i],np.array([sino['Ns'],num_angles]),0),dtype=afnp.complex64)
#  Ax = padmat_v2(norm_data[i],np.array([sino['Ns'],num_angles]),0,temp_mat)
#  temp_mat=temp_mat*0
  #filtered back-projection
  y = back_project(Ax,params)
  
elapsed_time = (time.time()-t)
print('Time for Back-proj of all slices :',elapsed_time)


plt.imshow(np.abs(y),cmap='gray');plt.colorbar();plt.title('Reconstructed slice');plt.show()
