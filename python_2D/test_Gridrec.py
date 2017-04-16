import tomopy
import time
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from tomoCam import gpuGridrec

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
pg.image(obj);pg.QtGui.QApplication.exec_()
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
input_params['fbp_filter_param']=0.5
t=time.time()
rec_gridrec = gpuGridrec(tomo,theta,sino_center,input_params)
elapsed_time = (time.time()-t)
print('Time for reconstucting using GPU-Gridrec of %d slices: %f' % (num_slice,elapsed_time))
pg.image(rec_gridrec/(num_angles*im_size));pg.QtGui.QApplication.exec_()
