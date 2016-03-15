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

#include "polargrid.h"
#include "pyGnufft.h"

PyObject *cPolarSample(PyObject *self, PyObject *prhs) {

    PyObject *in0, *in1, *in2, *in3, *in4;
    float kernel_radius, kernel_lookup_table_scale;

    if (!(PyArg_ParseTuple(prhs, "OOOOOff", &in0, &in1, &in2,
        &in3, &in4, &kernel_lookup_table_scale, &kernel_radius))){
        return NULL;
    }

    // data POINTERS
    int size;
    af::array * d_sx = PyAfnumpy_AsArrayfireArray(in0, FLOAT32);//gxi
    af::array * d_sy = PyAfnumpy_AsArrayfireArray(in1, FLOAT32);//gyi
    af::array * d_gv = PyAfnumpy_AsArrayfireArray(in2, CMPLX32);//x
    af::array * grid_dims = PyAfnumpy_AsArrayfireArray(in3, INT32);//grid
    af::array * d_klut = PyAfnumpy_AsArrayfireArray(in4, FLOAT32);//kblut
    
    printf("%f %f\n",kernel_lookup_table_scale,kernel_radius);
    fflush(stdout);

    float * d_samples_x = d_sx->device<float>();
    float * d_samples_y = d_sy->device<float>();
    int npoints = d_sx->elements();
    af::cfloat * temp_grid_vals = d_gv->device<af::cfloat>();
    cusp::complex<float> * d_grid_values = reinterpret_cast<cusp::complex<float> *>(temp_grid_vals);
    float * d_kernel_lookup_table = d_klut->device<float>();
    int kernel_lookup_table_size = d_klut->elements();
    int * gdims = grid_dims->host<int>();

    uint2 grid_size = { gdims[0], gdims[1] };
    cusp::complex<float> *d_samples_values;
    cuda_sample(d_samples_x, d_samples_y, d_grid_values, npoints, grid_size,
                d_kernel_lookup_table, kernel_lookup_table_size,
                kernel_lookup_table_scale, kernel_radius, d_samples_values);

    // GET OUTPUT
    int nd = 1;
    int dims[] = { npoints, 0 }; 
    PyObject * Svals = PyAfnumpy_Array(nd, dims, CMPLX32, d_samples_values, true);
    delete d_sx;
    delete d_sy;
    delete d_gv;
    delete d_klut;
    delete grid_dims;
    return Svals;
}
