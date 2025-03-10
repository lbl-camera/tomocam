import numpy as np
from . import cTomocam


def gradients(model, sinoT, psf):
    """Computes gradients of difference between forward projection of the model and the data

    Parameters
    -----------
    model: numpy.ndarray
        3-D model on obejct (single precision)
    sinoT: numpy.ndarray
        Backprojection of the sinogram (single precision)
    psf: numpy.ndarray
        Point spread function (single precision)

    Returns
    --------
    numpy.ndarray
        Gradients of the difference between forward projection of the model and the data
    """

    # cast the data to single precision
    model = model.astype(np.float32)
    sinoT = sinoT.astype(np.float32)
    psf = psf.astype(np.float32)

    return cTomocam.gradients(model, sinogram, psf)

