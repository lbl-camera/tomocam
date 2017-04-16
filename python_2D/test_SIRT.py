import tomopy
import time
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from tomoCam import gpuSIRT

num_slice = 50
im_size = 512 #2560 #A n X n X n volume
sino_center = im_size/2#1280
num_angles = 512 #1024
gpu_device = 2
oversamp_factor=1.5
num_iter = 150
p=1.2
sigma=.1



obj = tomopy.shepp3d((num_slice,im_size,im_size)) # Generate an object.
theta = tomopy.angles(num_angles) # Generate uniformly spaced tilt angles.
### Comparing to tomopy 
tomo = tomopy.project(obj,theta)
proj_dim = tomo.shape[2]
tomo= tomo[:,:,proj_dim/2-im_size/2:proj_dim/2+im_size/2]
pg.image(tomo);pg.QtGui.QApplication.exec_()
################## GPU MBIR ######################
input_params={}
input_params['gpu_device']=gpu_device
input_params['oversamp_factor']=oversamp_factor
input_params['num_iter']=num_iter
input_params['p']=p
input_params['smoothness']=sigma
t=time.time()
rec_sirt = gpuSIRT(tomo,theta,sino_center,input_params)
elapsed_time = (time.time()-t)
print('Time for reconstucting using GPU-SIRT of %d slices with %d iter : %f' % (num_slice,num_iter,elapsed_time))
pg.image(rec_sirt);pg.QtGui.QApplication.exec_()
