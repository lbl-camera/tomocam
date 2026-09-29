import numpy as np


def as_array(array, name):
    """Cast input to a C-contiguous, single-precision 2-D or 3-D array.

    The C++ layer treats a 2-D array as a single slice, i.e. (n, m) is
    interpreted as (1, n, m).
    """
    arr = np.ascontiguousarray(array, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(
            '{} must be 2-D or 3-D, got {}-D'.format(name, arr.ndim))
    return arr


def as_angles(angles):
    """Cast projection angles to a contiguous, single-precision 1-D array."""
    arr = np.ascontiguousarray(angles, dtype=np.float32).ravel()
    if arr.size == 0:
        raise ValueError('angles must not be empty')
    return arr


def check_nangles(sinogram, angles):
    """Check that the sinogram's projection axis matches the angles."""
    nangles = sinogram.shape[-2]
    if angles.size != nangles:
        raise ValueError(
            'number of angles ({}) does not match the projection axis of the '
            'sinogram ({})'.format(angles.size, nangles))
