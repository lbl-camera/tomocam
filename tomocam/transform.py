import numpy as np
from . import cTomocam

def radon(volume, angles, center):
    """Computes the radon transform using nufft.

    Parameters
    -----------
    volume: numpy.ndarray
        Data for which radon transform is seeketh, (single precision)
    angles: numpy.ndarray
        Projection angles (single, precision)
    center: float
        Offset correction to be applied to center of rotation

    Returns
    --------
        numpy.ndarray
            Radon transform of input volume
    """
    if volume.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')

    # compute transformation
    return cTomocam.radon(volume, angles, center, over_sample)


def backproject(sinogram, angles, center):
    """Computes the back-projection transform using nufft.

    Parameters
    ----------
    sinogram: numpy.ndarray 
        Projection data for which radon transform is seeketh, (single precision)
    angles: numpy.ndarray
        Projection angles (single, precision)
    center: float
        Offset correction to be applied to center of rotation

    Returns
    --------
        numpy.ndarray
            Inverse radon transform of input projection data
    """
    if sinogram.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')
   
    # create appropriate data-structures 
    return cTomocam.radon_adj(sinogram, angles, center, over_sample)
