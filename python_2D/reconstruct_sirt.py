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
from tomoCam import gpuSIRT
#from XT_ForwardModel import forward_project, init_nufft_params, back_project

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
        input_params={}
        input_params['gpu_device']=inputs['gpu_device']
        input_params['oversamp_factor']=oversamp_factor
        input_params['num_iter']=num_iter
        rec_sirt_final = gpuSIRT(tomo,theta,inputs['rot_center'],input_params)
        pg.image(rec_sirt_final);pg.QtGui.QApplication.exec_()

        ##################TomoPy Recon#####################
#        print('Recon - tomopy ASTRA-SIRT')
#        options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':num_iter}
#        t=time.time()
#        rec_tomopy = tomopy.recon(tomo, theta, center=inputs['rot_center'],algorithm=tomopy.astra,options=options)#emission=False)
#        elapsed_time = (time.time()-t)
#        print('Time for reconstucting using Tomopy SIRT of %d slices : %f' % (num_slice,elapsed_time))
        
#        fig = plt.figure()
#        sirt_Tomopy = np.flipud(rec_tomopy[0])
#        plt.imshow(sirt_Tomopy,cmap=plt.cm.Greys_r,vmin=0,vmax=0.00075)
#        plt.colorbar()
#        fig.suptitle('Tomopy SIRT-ASTRA Reconstruction')
#        plt.show()
#        plt.draw()

#        fig,ax=plt.subplots()
#        ax.plot(sirt_camTomo[sirt_camTomo.shape[0]//2],'r',label='NUFFT-SIRT')
#        ax.plot(sirt_Tomopy[sirt_Tomopy.shape[0]//2],'b',label='TomoPy-SIRT')
#        ax.legend()
#        plt.draw()

#        plt.show()

        pg.image(rec_tomopy);pg.QtGui.QApplication.exec_()

        print 'main: Done!'
main()


