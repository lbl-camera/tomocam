import numpy as np
from . import cTomocam


def calc_gradients(model, sinogram, angles, center=0, over_sample=2):
    """Computes gradients by taking iradon transform of difference between radon tranform of model and projection data

    Parameters
    -----------
    model: numpy.ndarray
        3-D model on obejct (single precision)
    sinogram: numpy.ndarray
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
    cTomocam.gradients(model, sinogram, angles, center, over_sample)


def update_total_variation(model, gradients, smoothness=1.0E-03):
    """Add constraints to gradients in-place

    Parameters:
    -----------

    model: numpy.ndarray
        model of the volume beign scanned (single precision)
    gradients: numpy.ndarray
        gradients of the error between mdoel and data
    smoothness: scalar, default = 0.001
    """
    if model.dtype != np.float32 and gradients.dtype != np.float32:
        raise ValueError('input data must be single precision')
    cTomocam.total_variation(gradients, model, smoothness) 


def MBIR(sinogram, angles, center, num_iters = 100, over_sample=1.5, smoothness=1.0E-03):
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
    smoothness: float (>= 0)
        Controls smoothness of reconstruction

    Returns
    --------
        numpy.ndarray
            Reconstructed tomographic volume
    """
    if sinogram.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')

    if smoothness <= 0:
        raise ValueError('smoothness value must be greater than 0')
    sigma = 1./smoothness

    return cTomocam.mbir(sinogram, angles, center, num_iters, over_sample, sigma)
