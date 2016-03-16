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
    print x1.shape
    #plt.imshow(afnp.real(x1));plt.show();
    x1 = x1.astype(np.complex64)
    x2 = gnufft.polarsample(params['gxi'],params['gyi'],x1,params['grid'],params['gkblut'],params['scale'],params['k_r']); #Fourier space to polar coordinates interpolation (qxy to qt)
    print x2.shape
    x3 = params['fftshift1D'](af_fft.ifft(params['fftshift1D_center'](x2)))*params['sino_mask'] #Polar cordinates to real space qt to rt 
    return x3 

def back_project(y,params):
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 

    y1 = params['fftshift1D_center'](afnp.fft.fft(params['fftshift1D'](y)))
    y2 = gnufft.polargrid_cub(params['gxi'],params['gyi'],y2,params['grid'],params['gs_per_b'],params['gb_dim_x'],params['gb_dim_y'],params['gs_in_bin'],params['gb_offset'],params['gb_loc'],params['gb_points_x'],params['gb_points_y'],params['gkblut'],params['scale'])
    y3 = params['fftshift2D'](af_fft.ifft2(y2*params['fftshift2D']))*params['deapod_filt']*params['Ns']
    return y3 

def init_nufft_params(sino,geom):
    #inputs : sino - A list contating parameters associated with the sinogram 
    #              Ns : Number of entries in the padded sinogram along the "detector" rows 
    #              Ns_orig :  Number of entries  detector elements per slice
    #              center : Center of rotation in pixels computed from the left end of the detector
    #              angles : An array containg the angles at which the data was acquired in radians
    #       : geom - TBD
    #
    
    KBLUT_LENGTH = 256;
    SCALING_FACTOR = 1.7;#What is this ? 
    k_r=3 #kernel size 2*kr+1
    beta =2*math.pi*2  
    Ns = sino['Ns']
    Ns_orig = sino['Ns_orig']
    ang = sino['angles']

    q_grid = range(1,sino['Ns']+1) - np.floor((sino['Ns']+1)/2) - 1
    sino['tt'],sino['qq']=np.meshgrid(ang*180/math.pi,q_grid)

    # Preload the Bessel kernel (real components!)
    kblut,KB,KB1D,KB2D=KBlut(k_r,beta,KBLUT_LENGTH) 
    KBnorm=np.array(np.single(np.sum(np.sum(KB2D(np.reshape(np.array(range(-k_r,k_r+1)),(2*k_r+1,1)),(np.array(range(-k_r,k_r+1))))))))
    print KBnorm
    kblut=kblut/KBnorm*SCALING_FACTOR #scaling fudge factor

################# Forward projector params #######################

    # polar to cartesian, centered
    [xi,yi]=pol2cart(sino['qq'],sino['tt']*math.pi/180)
    xi = xi+np.floor((Ns+1)/2)
    yi = yi+np.floor((Ns+1)/2)
   
    params={}
    params['k_r'] = k_r;
    params['deapod_filt']=afnp.array(deapodization(Ns,KB,Ns_orig))
    params['sino_mask'] = afnp.array(padmat(np.ones((Ns_orig,sino['qq'].shape[1])),np.array((Ns,sino['qq'].shape[1])),0))
    params['grid'] = afnp.array([Ns,Ns],dtype=np.int32)
    params['scale']= ((KBLUT_LENGTH-1)/k_r)
    params['center'] = sino['center']
    params['Ns'] = Ns

    # push parameters to gpu
    params['gxi'] = afnp.array(np.single(xi))
    params['gyi'] = afnp.array(np.single(yi))
    params['gkblut'] = afnp.array(np.single(kblut))
    params['det_grid'] = afnp.reshape(afnp.array(range(0,sino['Ns'])),(Ns,1)) 
    params['fft2Dshift'] = afnp.array(((-1)**params['det_grid'])*((-1)**params['det_grid'].T))
    params['fftshift1D'] = lambda x : ((-1)**params['det_grid'])*x
    params['fftshift1D_center'] = lambda x : afnp.exp(-1j*2*params['center']*(afnp.pi/params['Ns'])*params['det_grid'])*x
    params['fftshift1Dinv_center'] = lambda x : afnp.exp(1j*2*params['center']*afnp.pi/params['Ns']*params['det_grid'])*x

################# Back projector params #######################
    #[s_per_b,b_dim_x,b_dim_y,s_in_bin,b_offset,b_loc,b_points_x,b_points_y] = gnufft.polarbin1(xi,yi,params['grid'],4096*4,k_r);
    #params['gs_per_b']=afnp.array(s_per_b,dtype=afnp.int64) #int64
    #params['gs_in_bin=afnp.array(s_in_bin,dtype=afnp.int64);
    #params['gb_dim_x']= afnp.array(b_dim_x,dtype=afnp.int64);
    #params['gb_dim_y']= afnp.array(b_dim_y,dtype=afnp.int64);
    #params['gb_offset']=afnp.array(b_offset,dtype=afnp.int64);
    #params['gb_loc']=afnp.array(b_loc,dtype=afnp.int64);
    #params['gb_points_x']=afnp.array(b_points_x,dtype=afnp.float32);
    #params['gb_points_y']=afnp.array(b_points_y,,dtype=afnp.float32);


    return params

def deapodization(Ns,KB,Ns_orig):

    xx=np.array(range(1,Ns+1))-Ns/2-1
    dpz=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.reshape(KB(xx,np.array(0)),(np.size(xx),1))*KB(xx,np.array(0)))))
    # assume oversampling, do not divide outside box in real space:
    msk = padmat(np.ones((Ns_orig,Ns_orig)),np.array((Ns,Ns)),0)
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
