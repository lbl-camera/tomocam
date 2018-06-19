import gnufft 
import arrayfire as af
from .fft import fftshift, ifftshift
from .utils import multiply

def _forward_project(slcs, params):
    """
    Transpose a set of voxels into sinogram space. Not intended to be called 
    fom outside of tomocam.

    Parameters:
    -----------
    slcs: af.Array, 
        Set of voxels i.e. 3-D volume, ordeing [n_slice, rows, cols]
    params: dict,
        Recon. parameters.

    Returns:
    --------
    af.Array,
        3-D array of sinograms
    """

    # Fourier transform of slices
    qt = af.fft2(fftshift(multiply(slcs, params['ApodFilter'])))
    
    # Resample from a polar-grid 
    qt = gnufft.polarsample(params['Gxy'], qt, params['gKBLUT'], params['Scale'], params['KR'])

    # Inverse Fourier transform of resmapled data
    return multiply(af.ifft(fftshift(qt, params['Center'])).real(), params['sino_mask'])

def _backward_project(sino, params):
    """
    Transpose a set of sinograms into voxel space. Not intended to called from
    outside of tomogram.

    Parameters:
    -----------
    sino: af.Array,     
        sinograms, ordeing: [n_slice, n_pixel, n_angles]
    params: dict,
        Recon. parameters.

    Returns:
    --------
    af.Array,
        reconstructed volume 
    """

    # Fourier transform the sinogram
    qt = params['giDq'] * af.fft(fftshift(sino, center=params['Center']))

    # Resample on a polar-grid
    qt = gnufft.polarsample_transpose(params['Gxy'], qt, params['Grid'], params['gKBLUT'],\
             params['Scale'], params['KR'])

    # Inverse Fourier transform of the polar grid
    return multiply(af.ifft2(fftshift(qt)).real() * params['ApodFilter'])
