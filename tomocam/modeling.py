import numpy as np
from . import cTomocam


def calc_gradients(model, sinogram, angles, center=0, over_sample=2):
    """Computes gradients by taking iradon transform of difference between radon tranform of model and projection data

    Parameters
    -----------
    model: tomocam.DistArray
        3-D model on obejct (single precision)
    sinogram: tomocam.DistArray
        Projection data in sinogram form (single precision)
    angles: numpy.ndarray
        Projection angles (single precision)
    center: float
        Offset correction to be applied to center of rotation
    over_sample: float
        Zero padding to be added to signal
    """

    if model.dtype != np.float32 and data.dtype != np.float32 and angles.data != np.float32:
        raise ValueError('input data must be single precision')
    cTomocam.gradients(model.handle, sinogram.handle, angles, center, over_sample)


def update_total_variation(model, gradients, p=1.2, smoothness=0.1):
    """Add constraints to gradients in-place

    Parameters:
    -----------

    model: tomocam.DistArray
        model of the volume beign scanned (single precision)
    gradients: tomocam.DistArray
        gradients of the error between mdoel and data
    p: scalar, hyperparamter, default = 1.2
    smoothness: scalar, default = 0.1
    """
    if model.dtype != np.float32 and gradients.dtype != np.float32:
        raise ValueError('input data must be single precision')
    cTomocam.total_variation(gradients.handle, model.handle, p, smoothness) 


def MBIR(sinogram, angles, center, num_iters = 100, over_sample=1.5, smoothness=1.0):
    """Computes the Model-based Iterative reconstruction using nufft.

    Parameters
    ----------
    sinogram: numpy.ndarray
        Projection data for which radon transform is seeketh, (single precision)
    angles: numpy.ndarray
        Projection angles (single, precision)
    center: float
       Center of rotation
    num_iters: int
        Number of iterations
    over_sample: float
        Zero padding to be added to signal for fft
    smoothness: float
        Controls smoothness of reconstruction

    Returns
    --------
        numpy.ndarray
            Reconstructed tomographic volume
    """
    if sinogram.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')

    return cTomocam.mbir(sinogram, angles, center, num_iters, over_sample, smoothness)
