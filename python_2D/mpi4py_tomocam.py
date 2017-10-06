#!/usr/bin/env python

from __future__ import print_function
from mpi4py import MPI
import numpy as np
from tomoCam import gpuGridrec, gpuMBIR
import pyqtgraph as pg
import tomopy

def reconSIRT_tomocam(proj_data,angles,gpu_index):
    NUM_ITER=100
    input_params={}
    input_params['gpu_device']=gpu_index
    input_params['oversamp_factor']=1.5
    input_params['num_iter']=NUM_ITER
    num_ang,num_row,num_col=proj_data.shape
    center = num_col/2
    vol = gpuSIRT(proj_data,angles,center,input_params)
    return vol

def reconMBIR_tomocam(proj_data,angles,gpu_index):
    NUM_ITER=100
    input_params={}
    input_params['gpu_device']=gpu_index
    input_params['oversamp_factor']=1.5
    input_params['num_iter']=NUM_ITER
    input_params['p']=1.2
    input_params['smoothness']=.5
    num_ang,num_row,num_col=proj_data.shape
    center = num_col/2
    vol = gpuMBIR(proj_data,angles,center,input_params)
    return vol

comm = MPI.COMM_WORLD
print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
comm.Barrier()   # wait for everybody to synchronize _here_

num_slice = 512
num_col = 2560
num_angles = 1024
blk_size=np.int(num_slice/comm.size)
ang = tomopy.angles(num_angles) # Generate uniformly spaced tilt angles.
obj_recv=np.zeros(blk_size*num_col*num_col,dtype=np.float32)

if(comm.rank == 0):
#    obj = np.random.rand(num_slice,num_col,num_col).astype(np.float32)
    obj=np.float32(tomopy.shepp3d((num_slice,num_col,num_col))) # Generate an object.
    vol=np.zeros((num_slice,num_col,num_col),dtype=np.float32)
else:
    obj = None
    chunks = None
    vol=np.empty(num_slice*num_col*num_col)

comm.Scatter(obj,obj_recv,root=0)
#proj_data = np.random.rand(ang.size,num_slice,num_col) # Calculate projections for the relevant chunk
proj_data = tomopy.project(obj_recv.reshape(blk_size,num_col,num_col), ang) # Calculate projections for the relevant chunk
theta,rows,cols = proj_data.shape
proj_data = proj_data[:,:,cols/2-num_col/2:cols/2+num_col/2]

t_start = MPI.Wtime()
rec = np.float32(reconMBIR_tomocam(proj_data,ang,comm.rank))
t_diff = MPI.Wtime()-t_start
print('Time taken by process %d : %f s' %(comm.rank,t_diff))

# gather reconstructed volumes
comm.Gather(rec,vol,root=0)

if(comm.rank==0):
    vol=vol.reshape(num_slice,num_col,num_col)
    pg.image(vol);pg.QtGui.QApplication.exec_()
