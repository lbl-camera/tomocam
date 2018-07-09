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
    af::array afPtPos = PyAF_GetArray(pyPtPos);
    complex_t * point_pos = (complex_t *) afPtPos.device<af::cfloat>();
    int npoints = afPtPos.elements();

    // grid values 
    af::array afGridVals = PyAF_GetArray(pyGridVals);
    complex_t * grid_values = (complex_t *) afGridVals.device<af::cfloat>();
    int dims[2] = {(int) afGridVals.dims(0), (int) afGridVals.dims(1) };
    uint2 grid_size = { (unsigned) dims[0], (unsigned) dims[1] };

    // Kaiser Bessel Lookup Table
    af::array afKernelLUT = PyAF_GetArray(pyKernelLUT);
    float  * kernel_lookup_table = afKernelLUT.device<float>();
    int kernel_lookup_table_size = afKernelLUT.elements();

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
    return PyAF_FromData(nd, dims, FLOAT32, samples_values);
}
