from . import cTomocam
from ._utils import as_angles, as_array, check_nangles


def _sigma(smoothness):
    if smoothness <= 0:
        raise ValueError('smoothness value must be greater than 0')
    return 1. / smoothness


def MBIR(sinogram, angles, center, num_iters=50, smoothness=0.01,
         tol=1.0E-04, xtol=1.0E-04):
    """Computes the Model-based Iterative reconstruction using nufft.

    Parameters
    ----------
    sinogram: numpy.ndarray
        Projection data, (nslices, nangles, ncols) or (nangles, ncols)
    angles: numpy.ndarray
        Projection angles, in radians
    center: float
       Center of rotation, in pixels along the detector axis
    num_iters: int
        Maximum number of iterations
    smoothness: float (> 0)
        Controls smoothness of reconstruction
    tol: float
        Value of objective function at which to stop iteration
    xtol: float
        Value of change in solution at which to stop iteration

    Returns
    --------
        numpy.ndarray
            Reconstructed tomographic volume, (nslices, ncols, ncols)
    """
    sinogram = as_array(sinogram, 'sinogram')
    angles = as_angles(angles)
    check_nangles(sinogram, angles)

    num_iters = int(num_iters)
    if num_iters < 1:
        raise ValueError('num_iters must be greater than 0')

    return cTomocam.mbir(sinogram, angles, float(center), num_iters,
                         _sigma(smoothness), float(tol), float(xtol))


def MBIR_MPI(sinogram, angles, center, num_iters=50, smoothness=0.01,
             tol=1.0E-05, xtol=1.0E-05, file_write=False, output_file=""):
    """Computes the Model-based Iterative reconstruction using nufft with MPI support.

    Each rank passes its own slab of the sinogram; the reconstruction is
    gathered across ranks.

    Parameters
    ----------
    sinogram: numpy.ndarray
        Projection data, (nslices, nangles, ncols) or (nangles, ncols)
    angles: numpy.ndarray
        Projection angles, in radians
    center: float
       Center of rotation, in pixels along the detector axis
    num_iters: int
        Maximum number of iterations
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
            Reconstructed tomographic volume, (nslices, ncols, ncols)
    """
    sinogram = as_array(sinogram, 'sinogram')
    angles = as_angles(angles)
    check_nangles(sinogram, angles)

    num_iters = int(num_iters)
    if num_iters < 1:
        raise ValueError('num_iters must be greater than 0')

    if file_write and not output_file:
        raise ValueError('output_file must be set when file_write is True')

    return cTomocam.mbir_mpi(sinogram, angles, float(center), num_iters,
                             _sigma(smoothness), float(tol), float(xtol),
                             bool(file_write), str(output_file))
