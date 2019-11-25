/* -------------------------------------------------------------------------------
* Tomocam Copyright (c) 2018
*
* The Regents of the University of California, through Lawrence Berkeley National
* Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
*  Energy). All rights reserved.
*
* If you have questions about your rights to use or distribute this software,
* please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
*
* NOTICE. This Software was developed under funding from the U.S. Department of
* Energy and the U.S. Government consequently retains certain rights. As such, the
* U.S. Government has been granted for itself and others acting on its behalf a
* paid-up, nonexclusive, irrevocable, worldwide license in the Software to
* reproduce, distribute copies to the public, prepare derivative works, and
* perform publicly and display publicly, and to permit other to do so.
*---------------------------------------------------------------------------------
*/

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
    af::array afPtPos = PyAF_GetArray(pyPtPos);
    complex_t * point_pos = (complex_t *) afPtPos.device<af::cfloat>();
    int npoints = afPtPos.elements();

    // polar grid values
    af::array afSampleVals = PyAF_GetArray(pySampleVals);
    complex_t * sample_values = (complex_t *) afSampleVals.device<af::cfloat>();

    // Kaiser-Bessel kernel 
    af::array afKernelLUT = PyAF_GetArray(pyKernelLUT);
    float * kernel_lookup_table = afKernelLUT.device<float>();
    int kernel_lookup_table_size = afKernelLUT.elements();
    int dims[2] = {(int) afSampleVals.dims(1), (int) afSampleVals.dims(1) }; //This guy should be image_x * image_x, square shape
    uint2 grid_size = { (unsigned) dims[0], (unsigned) dims[1] };
    
#ifdef DEBUG
    printf("grid_size : { %d, %d }\n", dims[0], dims[1]);
#endif

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
    int ndims = 2;
    return PyAF_FromData(ndims, dims, CMPLX32, grid_values);
}
