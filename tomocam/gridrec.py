import numpy as np
import arrayfire as af

from .init import init_nufft_params
from .kernels import _backward_project
from .utils import np2af, af2np


def gridrec(tomo, angles, center, gpu_device=0,
            oversamp_factor=0.2, fbp_param=0.1):
    """
    Gridrec reconstruction using GPU based gridding

    Parameters:
    ----------
    tomo: np.ndarray, 
        sinogram, format := (slice, cols, angles)
    angles: np.ndarray, radians
        array of angles
    center: float
        center of rotation
    gpu_device: integer, optional, default = 0
        device id on a multi-gpu machine
    oversamp_factor: float, optional, default = 0.2
        factor by which to pad the image data for FFT
    fbp_param: float, optional, default  = 0.1
        A number between 0-1 for setting the filter cut-off for FBP
    verbose: bool, default = False
        print (mostly irrelevent) information to the scareen
    """

    # set gpu_device
    af.set_device(gpu_device)

    # dimensions
    n_slice, n_angles, img_size = tomo.shape

    # padding size
    padded = np.int16(img_size * (1 + oversamp_factor))

    # Initialize structures for NUFFT
    sino = {}
    sino['nPadded'] = padded
    sino['nImage'] = img_size  
    sino['Center'] = center + (padded - img_size)//2
    sino['Angles'] = angles
    sino['Filter'] = fbp_param

    # initialize NUFFT parameters
    nufft_params = init_nufft_params(sino)

    # allocate arrays and move data to gpu
    
    Ax = af.constant(0, padded, d1=n_angles, d2=n_slice, dtype=af.Dtype.f32)
    idx = slice((padded - img_size)/2, (padded + img_size)/2)
    Ax[idx,:,:] = np2af(tomo)

    # reconstruct on device
    recon = _backward_project(Ax, nufft_params)[:, idx, idx]

    # return numpy array
    return af2np(recon)
