from __future__ import print_function
import numpy as np
import tomopy
import time
import h5py
import afnumpy as afnp
import matplotlib.pyplot as plt

from XT_ForwardModel import forward_project, init_nufft_params, back_project
from XT_Common import padmat,padmat_v2


fpath='/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x_Full.mat'
start_slice=500
end_slice=561
pad_size = 3200 #Size of image for Fourier transform

f = h5py.File(fpath)
norm_data=f['norm_data']
norm_data=np.transpose(norm_data,(2,1,0))
norm_data=norm_data[start_slice:end_slice] #Crop data set

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
for i in range(1,num_slice):
  #pad data array and move it to GPU 
  Ax = afnp.array(padmat(norm_data[i],np.array([sino['Ns'],num_angles]),0),dtype=afnp.complex64)
#  Ax = padmat_v2(norm_data[i],np.array([sino['Ns'],num_angles]),0,temp_mat)
#  temp_mat=temp_mat*0
  #filtered back-projection
  y = back_project(Ax,params)
  
elapsed_time = (time.time()-t)
print('Time for NUFFT Back-proj of %d slices : %f' % (num_slice,elapsed_time))


plt.figure();plt.imshow(np.abs(y),cmap='gray');plt.colorbar();plt.title('Reconstructed slice using FastNUFFT');plt.draw();

##Tomopy
data = np.transpose(norm_data,(2,0,1))
data.astype(np.float32)
#np.zeros((1024, 50, 2560), dtype=np.float32)
t = time.time()
rec = tomopy.recon(data, ang,center=sino_center, algorithm='gridrec', ncore=12)
print('Time for tomopy gridrec of %d slices : %f' % (data.shape[1],time.time() - t))

print(rec.shape)

plt.figure();plt.imshow(np.flipud(np.abs(rec[-1])),cmap='gray');plt.colorbar();plt.title('Reconstructed slice using TomoPy');plt.draw();

plt.show()
