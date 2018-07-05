#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>
#include <cuda.h>

#include "pyGnufft.h"
#include "af_api.h"
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
    complex_t * point_pos = (complex_t *) PyAF_DevicePtr(pyPtPos);
    int npoints = PyAF_Size(pyPtPos);

    // polar grid values
    complex_t * sample_values = (complex_t *) PyAF_DevicePtr(pySampleVals);

    // Kaiser-Bessel kernel 
    float * kernel_lookup_table = (float *) PyAF_DevicePtr(pyKernelLUT);
    int kernel_lookup_table_size = PyAF_Size(pyKernelLUT);

    int dims[2]; 
    int ndims = 2;
    /* Grid Dimension is a list of ints */
    if (PyList_Check(pyGridDims)){
        if (ndims != (int) PyList_Size(pyGridDims)) {
            fprintf(stderr,"Error: incorrect number or dimensions for output grid.\n");
            return NULL;
        }
        dims[0] = PyInt_AsLong(PyList_GetItem(pyGridDims, (Py_ssize_t) 0));
        dims[1] = PyInt_AsLong(PyList_GetItem(pyGridDims, (Py_ssize_t) 1));
    } else {
        fprintf(stderr, "Error: datatype for grid-dims must be a list.\n");
        return NULL;
    }

#ifdef DEBUG
    printf("grid_size : { %d, %d }\n", dims[0], dims[1]);
#endif
    uint2 grid_size = { (unsigned) dims[0], (unsigned) dims[1] };

    // Output: Grid Values
    size_t len = dims[0] * dims[1];
    complex_t * grid_values = NULL;
    cudaMalloc((void **) &grid_values, sizeof(complex_t) * len);
    polarsample_transpose(point_pos, sample_values, npoints, grid_size,
                kernel_lookup_table, kernel_lookup_table_size,
                kernel_lookup_table_scale, kernel_radius, grid_values);

#ifdef DEBUG
    if (!grid_values) fprintf(stderr, "Error: failed to allocate memory.");
    complex_t disp[10];
    cudaMemcpy(disp, grid_values, sizeof(complex_t)*10, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) printf("(%f, %f)  ", disp[i].x, disp[i].y);
    printf("\n");    
#endif
    // GET OUTPUT
    PyObject * out = PyAF_FromData(ndims, dims, CMPLX32, grid_values);
    return out;
}
