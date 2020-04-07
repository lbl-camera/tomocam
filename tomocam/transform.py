import numpy as np
from . import cTomocam

def radon(volume, angles, center=0, over_sample=2):
    """Computes the radon transform using nufft.

    Parameters
    -----------
    volume: tomocam.DistArray
        Data for which radon transform is seeketh, (single precision)
    angles: numpy.ndarray
        Projection angles (single, precision)
    center: float
        Offset correction to be applied to center of rotation
    over_sample: float
        Zero padding to be added to signal

    Returns
    --------
        tomocam.DistArray
            Radon transform of input volume
    """
    if volume.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')
    slcs, _, ncol = volume.shape
    nangle = angles.shape[0]
    sino = np.zeros((slcs, nangle, ncol), dtype=np.float32)

    # create appropriate data-structures 
    sinogram = cTomocam.DArray(sino)

    # compute transformation
    cTomocam.radon(volume.handle, sinogram, angles, center, over_sample)
    return sino
    

def iradon(sinogram, angles, center=0, over_sample=2):
    """Computes the radon transform using nufft.

    Parameters
    ----------
    sinogram: tomocam.DistArray
        Projection data for which radon transform is seeketh, (single precision)
    angles: numpy.ndarray
        Projection angles (single, precision)
    center: float
        Offset correction to be applied to center of rotation
    over_sample: float
        Zero padding to be added to signal

    Returns
    --------
        tomocam.DistArray
            Inverse radon transform of input projection data
    """
    if sinogram.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')
    slcs, _, ncol = sinogram.shape
   
    # create appropriate data-structures 
    vol = np.zeros((slcs, ncol, ncol), dtype=np.float32)
    volume  = cTomocam.DArray(vol)

    cTomocam.iradon(sinogram.handle, volume, angles, center, over_sample)
    return vol
