import tomopy
import argparse
import time
import os
import numpy as np
import afnumpy as afnp
import matplotlib.pyplot as plt
import afnumpy as afnp
import math
#import pyqtgraph as pg
#from normalize import normalize_bo
#from XT_argsParser import bl832inputs_parser
from XT_ForwardModel import forward_project, init_nufft_params, back_project
from XT_Common import padmat

num_slice = 4
im_size = 300 #2560 #A n X n X n volume
sino_center = im_size/2#1280
num_angles = 1200 #1024
slice_idx = num_slice/2 - 1

obj = tomopy.shepp3d((num_slice,im_size,im_size)) # Generate an object.
#obj = tomopy.shepp3d(im_size) # Generate an object.
ang = tomopy.angles(num_angles) # Generate uniformly spaced tilt angles.

### Comparing to tomopy 
sim = tomopy.project(obj,ang)

sino={}
geom={}
sino['Ns'] = 1024 #3624#im_size*2 #Sinogram size after padding
sino['Ns_orig'] = im_size #size of original sinogram
sino['center'] = sino_center + (sino['Ns']/2 - sino['Ns_orig']/2)  #for padded sinogram
sino['angles'] = ang
sino['filter']=0.95

pad_size= sino['Ns']

params = init_nufft_params(sino,geom)

##Create a simulated object to test forward and back-projection routines
pad_idx = slice(sino['Ns']/2-sino['Ns_orig']/2,sino['Ns']/2+sino['Ns_orig']/2)

temp_x = afnp.zeros((sino['Ns'],sino['Ns']),dtype=afnp.complex64)
print(temp_x.shape)

Ax = afnp.zeros((num_slice/2,sino['Ns'],num_angles),dtype=afnp.complex64)
y = afnp.zeros((num_slice/2,sino['Ns_orig'],sino['Ns_orig']),dtype=afnp.complex64)
t=time.time()
slice_1=slice(0,num_slice,2)
slice_2=slice(1,num_slice,2)
x=afnp.array(obj[slice_1]+1j*obj[slice_2],dtype=afnp.complex64)
print(x.shape)
for i in range(num_slice/2):
  temp_x[pad_idx,pad_idx]=x[i]
  Ax[i] = forward_project(temp_x,params) #(math.pi/2)*sino['Ns']
  y[i] = (back_project(Ax[i],params)[pad_idx,pad_idx]) # 
elapsed_time = (time.time()-t)
print('Time for Forward/Back-proj for %d slices: %f'% (num_slice,elapsed_time))
#x = afnp.array(padmat(obj[384],np.array([sino['Ns'],sino['Ns']]),0),dtype=afnp.complex64)

plt.figure()
plt.imshow(y[num_slice/4].imag,cmap='gray')
plt.colorbar()
plt.title('NUFFT back-proj')
plt.draw()

plt.figure();plt.imshow(obj[slice_idx],cmap='gray');plt.colorbar();
plt.title('Original slice');
plt.draw();
#plt.imshow(x,cmap='gray')

#####Plotting #######

#tomopy
tomopy_sim_slice = np.flipud(np.fliplr(padmat(sim[:,slice_idx,:],np.array([num_angles, sino['Ns']]),0)))
plt.figure();plt.imshow(tomopy_sim_slice,cmap='gray');plt.title('Tomopy projection');plt.colorbar();plt.draw();


Ax_error = np.array(Ax[num_slice/4].imag.T) - tomopy_sim_slice
Ax_rmse = np.sum(np.sum((Ax_error**2)/Ax_error.size))
print('Normalized RMSE = ', Ax_rmse)

plt.figure();plt.imshow(Ax[num_slice/4].imag.T,cmap='gray');plt.title('NUFFT projection');plt.colorbar();plt.draw();

plt.figure();plt.imshow(Ax_error,cmap='gray');plt.title('Difference in projection');plt.colorbar();plt.draw();


plt.show()


#plt.imshow(y,cmap='gray');plt.show()

