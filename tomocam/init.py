#! /usr/bin/env python

import numpy as np
import arrayfire as af
from scipy.special import iv
from .util import padmat, np2af

def init_nufft_params(sino):
    """
    Function to initialize parameters associated with the forward and backward model

    Parameters:
    -----------
        sino: dict
            parameters associated with the sinogram
        Ns: int 
            Number of entries in the padded sinogram along the "detector" rows
        Nx: int,  
            Number of entries  detector elements per slice
        center: float, 
            Center of rotation in pixels computed from the left end of the detector
        angles: np.ndarray, radians
            Array of angles at which the data was acquired 

    Returns:
    --------
        params: dict
            pre-computed parameters for the forward/backward model
    """

    KBLUT_LENGTH = 256
    kr = 3  # kernel size 2*kr+1 TODO:Breaks when kr is large. Why ?
    beta = 4*np.pi
    Ns = sino['nPadded']
    Nx = sino['nImage']
    angles = sino['Angles']
    n_angles = angles.shape[0]

    q_grid = np.arange(Ns) - Ns//2
    sino['tt'], sino['qq'] = np.meshgrid(angles, q_grid)

    # Preload the Bessel kernel (real components!)
    kblut, KB, KB1D, KB2D = KBlut(kr, beta, KBLUT_LENGTH)

    # polar to cartesian, Centered
    xi, yi = pol2cart(sino['qq'], sino['tt'])
    xi = xi + (Ns+1)//2
    yi = yi + (Ns+1)//2

    # push parameters to gpu 
    params = {}
    params['KR'] = kr
    params['ApodFilter'] = np2af(
        deApodization(Ns, KB1D).astype(np.float32))
    params['SinoMask'] = np2af(
        padmat(np.ones((Nx, sino['qq'].shape[1])), (Ns, sino['qq'].shape[1])))
    params['Grid'] = [Ns, Ns]
    params['Scale'] = (KBLUT_LENGTH-1)/kr
    params['Center'] = sino['Center']
    params['Ns'] = Ns
    params['Ntheta'] = angles.size

    params['Gxi'] = np2af(np.single(xi))
    params['Gyi'] = np2af(np.single(yi))
    params['Gxy'] = params['Gxi'] + 1j*params['Gyi']
    params['gKBLUT'] = np2af(np.single(kblut))

    params['DetGrid'] = np.arange(sino['nPadded'])[:,np.newaxis]
    temp_mask = np.ones(Ns, dtype=np.float32)
    kernel = np.ones(Ns, dtype=np.float32)
    if 'Filter' in sino:
        temp = np.linspace(-1, 1, Ns)
        kernel = (Ns)*np.fabs(temp)*np.sinc(temp/2)
        temp_pos = (1-sino['Filter'])/2
        temp_mask[0:np.int16(temp_pos*Ns)] = 0
        temp_mask[np.int16((1-temp_pos)*Ns):] = 0
    params['giDq'] = np2af(kernel*temp_mask).T
    return params


def deApodization(Ns, KB1D):
    x = KB1D(np.arange(Ns) - Ns//2)
    dpz = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.outer(x,x))))
    return 1./dpz.real

def KBlut(kr, beta, nlut):
    kk = np.linspace(0, kr, nlut)
    kblut = KB2(kk, 2*kr, beta)
    scale = (nlut-1)/kr
    def kbcrop(x): return (np.abs(x) <= kr)
    def KBI(x): return np.int16(np.abs(x)*scale-np.floor(np.abs(x)*scale))
    def KB1D(x): return (np.reshape(kblut[np.int16(np.floor(np.abs(x)*scale)*kbcrop(x))], \
                    x.shape)*KBI(x)+ \
                    np.reshape(kblut[np.int16(np.ceil(np.abs(x)*scale)*kbcrop(x))],\
                    x.shape)*(1-KBI(x)))*kbcrop(x)
    def KB(x, y): return KB1D(x)*KB1D(y)
    def KB2D(x, y): return KB1D(x)*KB1D(y)
    return kblut, KB, KB1D, KB2D

def KB2(x, kr, beta):
    w = iv(0, beta*np.sqrt(1-(2*x/kr)**2))
    w = (w*(x <= kr))
    return w


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


if __name__ == '__main__':
    sino = {}
    sino['nPadded'] = 2457
    sino['nImage'] = 2048
    sino['Center'] = 1200.3
    sino['Angles'] = np.linspace(0, 2*np.pi, 2001)[:-1]
    params = init_nufft_params(sino)
