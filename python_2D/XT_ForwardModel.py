import gnufft 
import numpy as np
import afnumpy as afnp 
import afnumpy.fft as af_fft
import scipy.special as sc_spl #For bessel functions
import tomopy

def forward_project(x,params):
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 

    x1 = af_fft.fftshift(af_fft.fft2(x*params['deapod_filt']))
   # x2 = gnufft.polarsample(x1,....)
   x2 = gnufft.polarsample(params['gxi'],params['gyi'],x1,params['grid'],params['gkblut'],params['scale'],params['k_r']);
    x3 = af_fft.fftshift(af_fft.ifft(x2))*params['sino_mask']
    return x3 

def back_project(y,params):
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 

    y1 = afnp.fft.fftshift(afnp.fft.fft(y/params.deapod_filt))
   # x2 = gnufft.polargrid_cub(x1,....)
   # x3 = afnp.fft(x2)
    return x1 

def init_nufft_params(sino,geom):

    KBLUT_LENGTH = 256;
    SCALING_FACTOR = 1.7;#What is this ? 
    k_r=3 #kernel size 2*kr+1
    beta =2*pi*2  

    # Preload the Bessel kernel (real components!)
    kblut,KB,KB1,KB2D=KBlut(k_r,beta,KBLUT_LENGTH); 
    KBnorm=afnp.array(single(sum(sum(KB2D((-k_r:k_r),(-k_r:k_r))))));#transpose!
    kblut=kblut/KBnorm*SCALING_FACTOR; %scaling fudge factor

#################Forward projector params#######################

    # polar to cartesian, centered
    [xi,yi]=pol2cart(tt*pi/180,1*qq)
    xi = xi+floor((Ns+1)/2)
    yi = yi+floor((Ns+1)/2)

    params={}
    params['kr'] = kr;
    params['deapod_filt']=deapodization(Ns,KB,Nr_orig);
    params['sino_mask'] = afnp.array(padmat(ones(Nr_orig,size(qq,2)),[Ns size(qq,2)]))
    params['grid'] = [Ns,Ns]
    params['scale']= np.single((KBLUT_LENGTH-1)/k_r); 

    # push parameters to gpu
    params['gxi']=afnp.array(np.single(xi))
    params['gyi']=afnp.array(np.single(yi))
    params['gkblut']=afnp.array(np.single(kblut))

#################Back projector params#######################

    return params

def deapodization(Ns,KB,Nr_orig)

    xx=afnp.array(range(1,Ns))-Ns/2-1
    dpz=circshift(ifftshift(ifft2(fftshift(KB(xx,0)*KB(xx,0)))),[0 0])
    # assume oversampling, do not divide outside box in real space:
    msk = logical(padmat(ones(Nr_orig),[Ns Ns]))
    dpz = single(dpz)
    dpz(~msk) = 1            #keep the value outside box
    dpz=1./dpz               #deapodization factor truncated
    dpz=dpz/dpz(Ns/2+1,Ns/2+1) #scaling
    return dpz

    
def KBlut(k_r,beta,nlut):

    kk=afnp.linspace(0,k_r,nlut)
    kblut = KB2( kk, 2*k_r, beta)
    scale = (nlut-1)/k_r
    kbcrop=@(x) (abs(x)<=k_r)
    KBI=@(x) abs(x)*scale-floor(abs(x)*scale);
    KB1D=@(x) (reshape(kblut(floor(abs(x)*scale).*kbcrop(x)+1),size(x)).*KBI(x)+...
          reshape(kblut(ceil(abs(x)*scale).*kbcrop(x)+1),size(x)).*(1-KBI(x)))...
          .*kbcrop(x);
    KB=@(x,y) KB1D(x).*KB1D(y);
    KB2D=@(x,y) KB1D(x)*KB1D(y);
    return kblut, KB, KB1D,KB2D

def KB2(x, k_r, beta):
    w = sc_spl.iv(0, beta*sqrt(1-(2*x/k_r)^2)) 
    w=w/abs(sc_spl.iv(0, beta))
    w=(w.*(x<=k_r))
    return w

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
