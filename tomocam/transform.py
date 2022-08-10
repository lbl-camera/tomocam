import numpy as np
from . import cTomocam

def radon(volume, angles, center, over_sample=1.5):
    """Computes the radon transform using nufft.

    Parameters
    -----------
    volume: numpy.ndarray
        Data for which radon transform is seeketh, (single precision)
    angles: numpy.ndarray
        Projection angles (single, precision)
    center: float
        Offset correction to be applied to center of rotation
    over_sample: float
        Zero padding to be added to signal

    Returns
    --------
        numpy.ndarray
            Radon transform of input volume
    """
    if volume.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')

    # compute transformation
    return cTomocam.radon(volume, angles, center, over_sample)


def radon_adj(sinogram, angles, center, over_sample=1.5):
    """Computes the inverse-radon transform using nufft.

    Parameters
    ----------
    sinogram: numpy.ndarray 
        Projection data for which radon transform is seeketh, (single precision)
    angles: numpy.ndarray
        Projection angles (single, precision)
    center: float
        Offset correction to be applied to center of rotation
    over_sample: float
        Zero padding to be added to signal

    Returns
    --------
        numpy.ndarray
            Inverse radon transform of input projection data
    """
    if sinogram.dtype != np.float32 and angles.dtype != np.float32:
        raise ValueError('input data-type must be single precision')
   
    # create appropriate data-structures 
    return cTomocam.radon_adj(sinogram, angles, center, over_sample)
