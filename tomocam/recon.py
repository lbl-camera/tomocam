import numpy as np
from . import cTomocam


def MBIR(sinogram, angles, center, num_iters = 50, smoothness=0.01, tol=1.0E-04, xtol=1.0E-04):
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
    smoothness: float (> 0)
        Controls smoothness of reconstruction
    tol: float
        Value of objective function at which to stop iteration
    xtol: float
        Value of change in solution at which to stop iteration 

    Returns
    --------
        numpy.ndarray
            Reconstructed tomographic volume
    """
    if sinogram.dtype != np.float32:
        sinogram = sinogram.astype(np.float32)

    if angles.dtype != np.float32:
        angles = angles.astype(np.float32)

    if smoothness <= 0:
        raise ValueError('smoothness value must be greater than 0')
    sigma = 1./smoothness

    return cTomocam.mbir(sinogram, angles, center, num_iters, sigma, tol, xtol)


def MBIR_MPI(sinogram, angles, center, num_iters = 50, smoothness=0.01, tol=1.0E-04, xtol=1.0E-04, file_write=False, output_file=""):
    """Computes the Model-based Iterative reconstruction using nufft with MPI support.

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
    smoothness: float (> 0)
        Controls smoothness of reconstruction
    tol: float
        Value of objective function at which to stop iteration
    xtol: float
        Value of change in solution at which to stop iteration
    file_write: bool
        Whether to write output to file
    output_file: str
        Output file path if file_write is True

    Returns
    --------
        numpy.ndarray
            Reconstructed tomographic volume
    """
    if sinogram.dtype != np.float32:
        sinogram = sinogram.astype(np.float32)

    if angles.dtype != np.float32:
        angles = angles.astype(np.float32)

    if smoothness <= 0:
        raise ValueError('smoothness value must be greater than 0')
    sigma = 1./smoothness

    return cTomocam.mbir_mpi(sinogram, angles, center, num_iters, sigma, tol, xtol, file_write, output_file)
