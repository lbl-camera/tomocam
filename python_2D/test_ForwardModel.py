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

from XT_argsParser import bl832inputs_parser
from XT_ForwardModel import forward_project, init_nufft_params, back_project
from XT_Common import padmat
from normalize import normalize_bo

num_slice = 50
im_size = 512 #A n X n X n volume
sino_center = im_size/2
num_angles = 256
slice_idx = num_slice/2

obj = tomopy.shepp3d((num_slice,im_size,im_size)) # Generate an object.
#obj = tomopy.shepp3d(im_size) # Generate an object.
ang = tomopy.angles(num_angles) # Generate uniformly spaced tilt angles.

### Comparing to tomopy 
sim = tomopy.project(obj,ang)

sino={}
geom={}
sino['Ns'] =768#3624#im_size*2 #Sinogram size after padding
sino['Ns_orig'] = im_size #size of original sinogram
sino['center'] = sino_center + (sino['Ns']/2 - sino['Ns_orig']/2)  #for padded sinogram
sino['angles'] = ang

params = init_nufft_params(sino,geom)

##Create a simulated object to test forward and back-projection routines
x = afnp.array(padmat(obj[slice_idx],np.array([sino['Ns'],sino['Ns']]),0),dtype=afnp.complex64)

t=time.time()
num_iter = 2               
for i in range(1,num_iter+1):
#Ax = (math.pi/2)*sino['Ns']*forward_project(x,params)
  Ax = forward_project(x,params)
elapsed_time = (time.time()-t)/num_iter
print('Time for Forward Proj :',elapsed_time)
print('Min value of real FP=%f' %(Ax.real.min()))
print('Max value of real FP = %f' %(Ax.real.max()))


t=time.time()
for i in range(1,num_iter+1):
  y = back_project(Ax,params)*sino['Ns_orig']*num_angles
elapsed_time = (time.time()-t)/num_iter
print('Time for Back Proj :',elapsed_time)
print('Min value of real BP=%f' %(y.real.min()))
print('Max value of real BP = %f' %(y.real.max()))


plt.imshow(np.abs(y),cmap='gray')
plt.colorbar()
plt.title('NUFFT back-proj')
plt.draw()
#plt.imshow(x,cmap='gray')

#####Plotting #######

#tomopy
tomopy_sim_slice = np.flipud(np.fliplr(padmat(sim[:,slice_idx,:],np.array([num_angles, sino['Ns']]),0)))
plt.figure();plt.imshow(tomopy_sim_slice,cmap='gray');plt.title('Tomopy projection');plt.colorbar();plt.draw();
#rec = tomopy.recon(sim, ang, algorithm='art') # Reconstruct object.


Ax_error = np.array(Ax.real.T) - tomopy_sim_slice
Ax_rmse = np.sum(np.sum((Ax_error**2)/Ax_error.size))
print('Normalized RMSE = ', Ax_rmse)

plt.figure();plt.imshow(Ax.real.T,cmap='gray');plt.title('NUFFT projection');plt.colorbar();plt.draw();

plt.figure();plt.imshow(Ax_error,cmap='gray');plt.title('Difference in projection');plt.colorbar();plt.draw();

plt.show()


#plt.imshow(y,cmap='gray');plt.show()
