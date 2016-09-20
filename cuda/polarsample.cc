#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <Python.h>

#include "pyGnufft.h"
#include "afnumpyapi.h"
#include "polarsample.h"

PyObject *cPolarSample(PyObject *self, PyObject *prhs) {
    PyObject *pyPtPos, *pyGridVals, *pyKernelLUT;
    float kernel_radius, kernel_lookup_table_scale;

    if (!(PyArg_ParseTuple(prhs, "OOOOOff", 
                    &pyPtPos, 
                    &pyGridVals, 
                    &pyKernelLUT, 
                    &kernel_lookup_table_scale, 
                    &kernel_radius))){
        return NULL;
    }

    // data POINTERS
    // point positions ( x = point_pos.x, y = point_pos.y )
    complex_t * point_pos = (complex_t *) PyAfnumpy_DevicePtr(pyPtPos);
    int npoints = PyAfnumpy_Size(pyPtPos);

    // grid values 
    complex_t * grid_values = (complex_t *) PyAfnumpy_DevicePtr(pyGridVals);
    int dims[2];
    dims[0] = PyAfnumpy_Dims(pyGridVals, 0);
    dims[1] = PyAfnumpy_Dims(pyGridVals, 1);
    uint2 grid_size = { (unsigned) dims[0], (unsigned) dims[1] };

    // Kaiser Bessel Lookup Table
    float * kernel_lookup_table = (float *) PyAfnumpy_DevicePtr(pyKernelLUT);
    int kernel_lookup_table_size = PyAfnumpy_Size(pyKernelLUT);

    // Output: Intesity values on polar-grid 
    complex_t * samples_values;
    cudaMalloc(&samples_values, sizeof(complex_t) * npoints);

    // do the interpolation on GPU
    polarsample(point_pos, grid_values, npoints, grid_size,
                kernel_lookup_table, kernel_lookup_table_size,
                kernel_lookup_table_scale, kernel_radius, samples_values);

    // GET OUTPUT
    int nd = 1;
    dims[0] = npoints;
    dims[1] = 1;
    PyObject * out = PyAfnumpy_FromData(nd, dims, CMPLX32, samples_values, true);
    return out;
}
