# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 08:53:10 2016

@author: lbluque
"""

import numpy as np


def _simple_normalize(tomo, flats, darks):

    denom = flats - darks
    denom[denom == 0] = 1e-6
    proj = np.zeros_like(tomo)
    weight = np.zeros_like(tomo)

    for m in range(tomo.shape[0]):
        weight[m,:,:] = tomo[m, :, :] - darks
        proj[m, :, :] = np.true_divide(weight[m,:,:], denom)

    return proj,weight


def normalize_bo(tomo, flats, darks, num_flat):
    """
    Normalize using the last set of flat fields only

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    flats : ndarray
        3D flats field data.
    dark : ndarray
        3D dark field data.
    num_flat : int
        number of flat field sets taken

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    flat_start = flats.shape[0]//num_flat
    tomo = np.array(tomo, dtype=np.float32)
    flats = np.array(flats[flat_start:], dtype=np.float32)
    dark = np.array(darks, dtype=np.float32)

    dark = np.mean(darks, axis=0)
    flats = np.mean(flats, axis=0)

    arr,weight = _simple_normalize(tomo, flats, dark)

    return arr,weight


def normalize_fo(tomo, flats, darks, num_flat):
    """
    Normalize using the first set of flat fields only

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    flats : ndarray
        3D flats field data.
    dark : ndarray
        3D dark field data.
    num_flat : int
        number of flat field sets taken

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    flat_end = flats.shape[0]//num_flat
    tomo = np.array(tomo, dtype=np.float32)
    flats = np.array(flats[:flat_end], dtype=np.float32)
    dark = np.array(darks, dtype=np.float32)

    dark = np.mean(darks, axis=0)
    flats = np.mean(flats, axis=0)

    arr,weight = _simple_normalize(tomo, flats, dark)

    return arr,weight


def normalize_fb(tomo, flats, darks):
    """
    Normalize using mean of all flat fields taken during experiment

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    flats : ndarray
        3D flats field data.
    dark : ndarray
        3D dark field data.


    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    tomo = np.array(tomo, dtype=np.float32)
    flats = np.array(flats, dtype=np.float32)
    dark = np.array(darks, dtype=np.float32)
    dark = np.mean(darks, axis=0)
    flats = np.mean(flats, axis=0)

    arr,weight = _simple_normalize(tomo, flats, dark)

    return arr,weight


def normalize_832(tomo, flats, darks, flat_loc,
                  cutoff=None, ncore=None, nchunk=None):
    """
    Normalize raw 3D projection data with flats taken more than once during
    tomography. Normalization for each projection is done with the mean of the
    nearest set of flats fields.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    flats : ndarray
        3D flats field data.
    dark : ndarray
        3D dark field data.
    flat_loc : list of int
        Indices of flats field data within tomography


    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """

    tomo = np.array(tomo, dtype=np.float32)  # dtype.as_float32(tomo)
    flats = np.array(flats, dtype=np.float32)  # dtype.as_float32(flats)
    dark = np.array(darks, dtype=np.float32)  # dtype.as_float32(dark)

    arr = np.zeros_like(tomo)

    dark = np.median(darks, axis=0)

    num_flats = len(flat_loc)
    total_flats = flats.shape[0]
    total_tomo = tomo.shape[0]
    num_per_flat = total_flats//num_flats  # should always be an integer
    tend = 0

    for m, loc in enumerate(flat_loc):
        fstart = m*num_per_flat
        fend = (m + 1)*num_per_flat
        flats = np.median(flats[fstart:fend], axis=0)

        tstart = 0 if m == 0 else tend
        tend = total_tomo if m >= num_flats-1 else (flat_loc[m+1]-loc)//2 + loc

        _arr = _simple_normalize(tomo[tstart:tend], flats, dark)

        arr[tstart:tend] = _arr

    return arr
