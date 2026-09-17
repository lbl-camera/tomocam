from . import cTomocam
from ._utils import as_angles, as_array, check_nangles


def radon(volume, angles):
    """Computes the radon transform using nufft.

    Parameters
    -----------
    volume: numpy.ndarray
        Volume to project, (nslices, nrows, ncols) or (nrows, ncols)
    angles: numpy.ndarray
        Projection angles, in radians

    Returns
    --------
        numpy.ndarray
            Radon transform of input volume, (nslices, nangles, ncols)

    Notes
    -----
    The forward projection is always computed about the center of the volume.
    Use the `center` argument of `backproject` to correct for an off-center
    axis of rotation on the way back.
    """
    volume = as_array(volume, 'volume')
    angles = as_angles(angles)

    # compute transformation
    return cTomocam.radon(volume, angles)


def backproject(sinogram, angles, center):
    """Computes the back-projection transform using nufft.

    Parameters
    ----------
    sinogram: numpy.ndarray
        Projection data, (nslices, nangles, ncols) or (nangles, ncols)
    angles: numpy.ndarray
        Projection angles, in radians
    center: float
        Center of rotation, in pixels along the detector axis

    Returns
    --------
        numpy.ndarray
            Inverse radon transform of input projection data,
            (nslices, ncols, ncols)
    """
    sinogram = as_array(sinogram, 'sinogram')
    angles = as_angles(angles)
    check_nangles(sinogram, angles)

    # the sinogram is padded internally, and the center adjusted to match
    return cTomocam.backproject(sinogram, angles, float(center))
