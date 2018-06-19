import gnufft 
import arrayfire as af
from .fft import fftshift, ifftshift
from .util import multiply

def _forward_project(slcs, params):
    """
  
 
    """
    # real space (rxy) to Fourier space (qxy) 
    qt = af.fft2(fftshift(multiply(slcs, params['ApodFilter'])))
    
    # resample from a polar-grid 
    qt = gnufft.polarsample(params['Gxy'], qt, params['gKBLUT'], params['Scale'], params['KR'])

    # Polar cordinates to real space qt to rt
    return multiply(af.ifft(fftshift(qt, params['Center'])).real(), params['sino_mask'])

def _backward_project(sino, params):
    """
    
   
    """
    # Fourier transform the sinogram
    qt = params['giDq'] * af.fft(fftshift(sino, center=params['Center']))

    # resample on a polar-grid
    qt = gnufft.polarsample_transpose(params['Gxy'], qt, params['Grid'], params['gKBLUT'],\
             params['Scale'], params['KR'])

    # Inverse Fourier transform of the polar grid
    return multiply(af.ifft2(fftshift(qt)).real() * params['ApodFilter'])
