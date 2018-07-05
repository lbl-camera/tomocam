#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#include "pyGnufft.h"
#include "af_api.h"
#include "polarsample.h"

PyObject *cPolarSample(PyObject *self, PyObject *prhs) {
    PyObject *pyPtPos, *pyGridVals, *pyKernelLUT;
    float kernel_radius, kernel_lookup_table_scale;

    if (!(PyArg_ParseTuple(prhs, "OOOff", 
                    &pyPtPos, 
                    &pyGridVals, 
                    &pyKernelLUT, 
                    &kernel_lookup_table_scale, 
                    &kernel_radius))){
        return NULL;
    }

    // data POINTERS
    // point positions ( x = point_pos.x, y = point_pos.y )
    complex_t * point_pos = (complex_t *) PyAF_DevicePtr(pyPtPos);
    int npoints = PyAF_Size(pyPtPos);

    // grid values 
    complex_t * grid_values = (complex_t *) PyAF_DevicePtr(pyGridVals);
    int dims[2];
    dims[0] = PyAF_Dims(pyGridVals, 0);
    dims[1] = PyAF_Dims(pyGridVals, 1);
    uint2 grid_size = { (unsigned) dims[0], (unsigned) dims[1] };

    // Kaiser Bessel Lookup Table
    float * kernel_lookup_table = (float *) PyAF_DevicePtr(pyKernelLUT);
    int kernel_lookup_table_size = PyAF_Size(pyKernelLUT);

#ifdef DEBUG
    /* check the values */
    fprintf(stderr, "kernel_lookup_table_size = %d.\n", kernel_lookup_table_size);
    float * kblut = (float *) malloc(sizeof(float) * kernel_lookup_table_size);
    cudaMemcpy(kblut, kernel_lookup_table, kernel_lookup_table_size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
        fprintf(stderr, "%f, ", kblut[i]); fflush(stderr);
    fprintf(stderr, "\n\n");
#endif // DEBUG


    // Output: Intesity values on polar-grid 
    complex_t * samples_values;
    cudaMalloc(&samples_values, sizeof(complex_t) * npoints);

    // do the interpolation on GPU
    polarsample(point_pos, grid_values, npoints, grid_size,
                kernel_lookup_table, kernel_lookup_table_size,
                kernel_lookup_table_scale, kernel_radius, samples_values);

#ifdef DEBUG
    /* check the values from kernel */
    fprintf(stderr, "output values: \n");
    complex_t disp[10];
    cudaMemcpy(disp, samples_values, sizeof(complex_t) * 10, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) fprintf(stderr, "(%f, %f), ", disp[i].x, disp[i].y);
    fprintf(stderr, "\n\n");
#endif

    // GET OUTPUT
    int nd = 2;
    dims[0] = PyAF_Dims(pyPtPos, 0);
    dims[1] = PyAF_Dims(pyPtPos, 1);
    PyObject * out = PyAF_FromData(nd, dims, CMPLX32, samples_values);
    return out;
}
