#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>
#include <cuda.h>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL gnufft_ARRAY_API
#include <numpy/arrayobject.h>

#include "afnumpyapi.h"
#include "pyGnufft.h"
#include "polarsample.h"

PyObject *cPolarSampleTranspose(PyObject *self, PyObject *prhs) {

    PyObject *pyPtPos, *pySampleVals, *pyKernelLUT, *pyGridDims;
    float kernel_radius, kernel_lookup_table_scale;

    if (!(PyArg_ParseTuple(prhs, "OOOOff", 
                    &pyPtPos, 
                    &pySampleVals, 
                    &pyGridDims, 
                    &pyKernelLUT, 
                    &kernel_lookup_table_scale, 
                    &kernel_radius))){
        return NULL;
    }

    // data POINTERS
    // point positions ( x = point_pos.x, y = point_pos.y )
    complex_t * point_pos = (complex_t *) PyAfnumpy_DevicePtr(pyPtPos);
    int npoints = PyAfnumpy_Size(pyPtPos);

    // polar grid values
    complex_t * sample_values = (complex_t *) PyAfnumpy_DevicePtr(pySampleVals);

    // Kaiser-Bessel kernel 
    float * kernel_lookup_table = (float *) PyAfnumpy_DevicePtr(pyKernelLUT);
    int kernel_lookup_table_size = PyAfnumpy_Size(pyKernelLUT);

    /* Grid Dimension is numpy array */
    int * dims = (int *) PyArray_DATA(pyGridDims);
    uint2 grid_size = { (unsigned) dims[0], (unsigned) dims[1] };

    // Output: Grid Values
    complex_t * grid_values;
    cudaMalloc((void **) &grid_values, sizeof(complex_t) * dims[0] * dims[1]);
    polarsample_transpose(point_pos, sample_values, npoints, grid_size,
                kernel_lookup_table, kernel_lookup_table_size,
                kernel_lookup_table_scale, kernel_radius, grid_values);

    // GET OUTPUT
    int nd = 2;
    PyObject * out = PyAfnumpy_FromData(nd, dims, CMPLX32, grid_values, true);
    return out;
}
