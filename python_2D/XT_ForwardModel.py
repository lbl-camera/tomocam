import gnufft 
import numpy as np
import afnumpy as afnp 

def forward_project(x,params):
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 
    print x 
    x1 = afnp.fft.fftshift(afnp.fft.fft2(x*params['deapod_filt']))
   # x2 = gnufft.polarsample(x1,....)
    x3 = afnp.fft.fftshift(afnp.fft.ifft2(x1))
    return x3 

def back_project(y,params)
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 

    y1 = afnp.fft.fftshift(afnp.fft.fft(y/params.deapod_filt))
   # x2 = gnufft.polarsample(x1,....)
   # x3 = afnp.fft(x2)
    return x1 

def init_nufft_params(sino,geom):
    params={}
    params.h=sino['N_t']
    return params

    
