#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <arrayfire.h>
#include <cusp/complex.h>
#include <cuda.h>
#include "cuda_sample.h"
#include <Python.h>
#include <numpy/arrayobject.h>

#include "afnumpyapi.h"
#include "polargrid.h"
#include "pyGnufft.h"

typedef cusp::complex<float> complex_t;

PyObject *cPolarSample(PyObject *self, PyObject *prhs) {

    PyObject *in0, *in1, *in2, *in4;
    PyArrayObject *in3;
    float kernel_radius, kernel_lookup_table_scale;

    if (!(PyArg_ParseTuple(prhs, "OOOOOff", &in0, &in1, &in2,
        &in3, &in4, &kernel_lookup_table_scale, &kernel_radius))){
        return NULL;
    }

    // data POINTERS
    float * d_samples_x = (float *) PyAfnumpy_DevicePtr(in0);
    float * d_samples_y = (float *) PyAfnumpy_DevicePtr(in1);
    int npoints = PyAfnumpy_Size(in0);
    int dims[2];
    dims[0] = PyAfnumpy_Dims(in0, 0);
    dims[1] = PyAfnumpy_Dims(in0, 1);

    complex_t * d_grid_values = (complex_t *) PyAfnumpy_DevicePtr(in2);
    float * d_kernel_lookup_table = (float *) PyAfnumpy_DevicePtr(in4);
    int kernel_lookup_table_size = PyAfnumpy_Size(in4);

    /* in3: Grid Dimension is numpy array */
    int * gdims = (int *) PyArray_DATA(in3);
    uint2 grid_size = { gdims[0], gdims[1] };

    complex_t * d_samples_values;
    cudaMalloc(&d_samples_values, sizeof(complex_t) * npoints);
    cuda_sample(d_samples_x, d_samples_y, d_grid_values, npoints, grid_size,
                d_kernel_lookup_table, kernel_lookup_table_size,
                kernel_lookup_table_scale, kernel_radius, d_samples_values);

    // GET OUTPUT
    int nd = 2;
    //af::cfloat * buf = reinterpret_cast<af::cfloat>(d_samples_values);
    PyObject * out = PyAfnumpy_FromData(nd, dims, CMPLX32, d_samples_values, true);
    return out;
}
