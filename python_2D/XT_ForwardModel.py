import gnufft 
import math
import numpy as np
import afnumpy as afnp 
import afnumpy.fft as af_fft
import scipy.special as sc_spl #For bessel functions
import tomopy
import matplotlib.pyplot as plt
from XT_Common import padmat

def forward_project(x,params):
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 
    x1 = params['fft2Dshift']*(af_fft.fft2(x*params['deapod_filt']*params['fft2Dshift']))/params['Ns'] #real space (rxy) to Fourier space (qxy)
    #plt.imshow(afnp.real(x1));plt.show();
    x1 = x1.astype(np.complex64)
    x2 = gnufft.polarsample(params['gxi'],params['gyi'],x1,params['grid'],params['gkblut'],params['scale'],params['k_r']); #Fourier space to polar coordinates interpolation (qxy to qt)
    x3 = params['fftshift1D'](af_fft.ifft(params['fftshift1D_center'](x2)))*params['sino_mask'] #Polar cordinates to real space qt to rt 
    return x3 

def back_project(y,params):
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 

    y1 = afnp.fft.fftshift(afnp.fft.fft(y/params.deapod_filt))
   # x2 = gnufft.polargrid_cub(x1,....)
   # x3 = afnp.fft(x2)
    return x1 

def init_nufft_params(sino,geom):
    #inputs : sino - 
    #       : geom - 
    
    KBLUT_LENGTH = 256;
    SCALING_FACTOR = 1.7;#What is this ? 
    k_r=3 #kernel size 2*kr+1
    beta =2*math.pi*2  
    Ns = sino['Ns']
    Nr_orig = sino['Nr_orig']
    qq = sino['qq']
    tt = sino['tt']

    # Preload the Bessel kernel (real components!)
    kblut,KB,KB1D,KB2D=KBlut(k_r,beta,KBLUT_LENGTH) 
    KBnorm=np.array(np.single(np.sum(np.sum(KB2D(np.reshape(np.array(range(-k_r,k_r+1)),(2*k_r+1,1)),(np.array(range(-k_r,k_r+1))))))))
    print KBnorm
    kblut=kblut/KBnorm*SCALING_FACTOR #scaling fudge factor

################# Forward projector params #######################

    # polar to cartesian, centered
    [xi,yi]=pol2cart(qq,tt*math.pi/180)
    xi = xi+np.floor((Ns+1)/2)
    yi = yi+np.floor((Ns+1)/2)

    params={}
    params['k_r'] = k_r;
    params['deapod_filt']=afnp.array(deapodization(Ns,KB,Nr_orig))
    params['sino_mask'] = afnp.array(padmat(np.ones((Nr_orig,qq.shape[1])),np.array((Ns,qq.shape[1])),0))
    params['grid'] = afnp.array([Ns,Ns],dtype=np.int32)
    params['scale']= ((KBLUT_LENGTH-1)/k_r)
    params['center'] = sino['center']
    params['Ns'] = Ns

    # push parameters to gpu
    params['gxi'] = afnp.array(np.single(xi))
    params['gyi'] = afnp.array(np.single(yi))
    params['gkblut'] = afnp.array(np.single(kblut))
    params['det_grid'] = np.reshape(np.array(range(0,sino['Ns'])),(Ns,1)) 
    params['fft2Dshift'] = afnp.array(((-1)**params['det_grid'])*((-1)**params['det_grid'].T))
    params['fftshift1D'] = lambda x : ((-1)**params['det_grid'])*x
    params['fftshift1D_center'] = lambda x : afnp.exp(-1j*2*params['center']*afnp.pi/params['Ns']*params['det_grid'])*x
    params['fftshift1Dinv_center'] = lambda x : afnp.exp(1j*2*params['center']*afnp.pi/params['Ns']*params['det_grid'])*x

################# Back projector params #######################

    return params

def deapodization(Ns,KB,Nr_orig):

    xx=np.array(range(1,Ns+1))-Ns/2-1
    dpz=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.reshape(KB(xx,np.array(0)),(np.size(xx),1))*KB(xx,np.array(0)))))
    # assume oversampling, do not divide outside box in real space:
    msk = padmat(np.ones((Nr_orig,Nr_orig)),np.array((Ns,Ns)),0)
    msk=msk.astype(bool)
    dpz=dpz.astype(float)
    dpz[~msk] = 1            #keep the value outside box
    dpz=1/dpz               #deapodization factor truncated
    dpz=dpz/dpz[Ns/2+1,Ns/2+1] #scaling
    return dpz

    
def KBlut(k_r,beta,nlut):
    kk=np.linspace(0,k_r,nlut)
    kblut = KB2( kk, 2*k_r, beta)
    scale = (nlut-1)/k_r
    kbcrop = lambda x: (np.abs(x)<=k_r)
    KBI = lambda x: np.int16(np.abs(x)*scale-np.floor(np.abs(x)*scale))
    KB1D = lambda x: (np.reshape(kblut[np.int16(np.floor(np.abs(x)*scale)*kbcrop(x))],x.shape)*KBI(x)+np.reshape(kblut[np.int16(np.ceil(np.abs(x)*scale)*kbcrop(x))],x.shape)*(1-KBI(x)))*kbcrop(x)
    KB=lambda x,y: KB1D(x)*KB1D(y)
    KB2D=lambda x,y: KB1D(x)*KB1D(y)
    return kblut, KB, KB1D,KB2D

def KB2(x, k_r, beta):
    w = sc_spl.iv(0, beta*np.sqrt(1-(2*x/k_r)**2)) 
    w=w/np.abs(sc_spl.iv(0, beta))
    w=(w*(x<=k_r))
    return w

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
