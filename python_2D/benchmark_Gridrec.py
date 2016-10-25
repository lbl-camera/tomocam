from __future__ import print_function
import numpy as np
import tomopy
import time
import h5py
import afnumpy as afnp
import arrayfire as af
import matplotlib.pyplot as plt

from XT_ForwardModel import forward_project, init_nufft_params, back_project
from XT_Common import padmat,padmat_v2

##Input parameters
fpath='/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x_Full.mat'
#'/home/svvenkatakrishnan/data/20131106_074854_S45L3_notch_OP_10x.mat'
af.set_device(7)
start_slice=300
end_slice=400
pad_size = 3200 #Size of image for Fourier transform
sino_center = 1294#1328
nufft_scaling = (np.pi/pad_size)**2
fbp_filter_param = 0.5 #Normalized number from 0-1

#Read and re-organize data
f = h5py.File(fpath)
norm_data=f['norm_data']
norm_data=np.transpose(norm_data,(2,1,0))
norm_data=norm_data[start_slice:end_slice] #Crop data set

num_slice =  norm_data.shape[0]
num_angles= norm_data.shape[2]
im_size =  norm_data.shape[1] #A n X n X n volume
slice_idx = 50#num_slice/2
ang = tomopy.angles(num_angles+1) # Generate uniformly spaced tilt angles.
ang=ang[:-1]

#Initialize structures for NUFFT
sino={}
geom={}
sino['Ns'] = pad_size #Sinogram size after padding
sino['Ns_orig'] = im_size #size of original sinogram
sino['center'] = sino_center + (sino['Ns']/2 - sino['Ns_orig']/2)  #for padded sinogram
sino['angles'] = ang
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
gdata=afnp.array(norm_data[slice_1]+1j*norm_data[slice_2],dtype=afnp.complex64)

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

##Tomopy
data = np.transpose(norm_data,(2,0,1))
data.astype(np.float32)
#np.zeros((1024, 50, 2560), dtype=np.float32)
t = time.time()
rec_tomopy = tomopy.recon(data, ang,center=sino_center, algorithm='gridrec', ncore=32)
print('Time for tomopy gridrec of %d slices : %f' % (data.shape[1],time.time() - t))


#Plotting results

if np.mod(slice_idx,2):
  nufft_slice = np.abs(rec_nufft[slice_idx/2].real)
else:
  nufft_slice = np.abs(rec_nufft[slice_idx//2].imag)
  
plt.figure();plt.imshow(nufft_slice,cmap='gray');plt.colorbar();plt.title('Reconstructed slice using FastNUFFT');plt.draw();

tomopy_slice = np.flipud(np.abs(rec_tomopy[slice_idx]))
plt.figure();plt.imshow(tomopy_slice,cmap='gray');plt.colorbar();plt.title('Reconstructed slice using TomoPy');plt.draw();

fig,ax=plt.subplots()
ax.plot(nufft_slice[nufft_slice.shape[1]//2],'r',label='NUFFT')
ax.plot(tomopy_slice[nufft_slice.shape[1]//2],'b',label='TomoPy')
ax.legend()
plt.draw()

plt.show()
