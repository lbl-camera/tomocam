#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <arrayfire.h>
#include <cuComplex.h>
#include <cusp/complex.h>
#include <cuda.h>
#include "cuda_sample.h"
#include <Python.h>
#include <numpy/arrayobject.h>

#include "pyGnufft.h"
#include "afnumpyapi.h"

//typedef cusp::complex<float> complex32_t;
typedef cuComplex complex32_t;


__global__ void makeComplex(float * real, float * imag, int npts, 
                       complex32_t * cplx){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < npts )
        //cplx[i] = complex32_t(real[i], imag[i]);
        cplx[i] = make_cuComplex(real[i], imag[i]);
}

PyObject *cDebug(PyObject *self, PyObject *prhs) {

    PyObject *in0, *in1;

    if (!(PyArg_ParseTuple(prhs, "OO", &in0, &in1))){
        return NULL;
    }

    // data POINTERS
    float * d_real = (float *) PyAfnumpy_DevicePtr(in0);
    float * d_imag = (float *) PyAfnumpy_DevicePtr(in1);
    int npoints = PyAfnumpy_Size(in0);
   
    complex32_t * d_cplx;
    cudaMalloc(&d_cplx, npoints * sizeof(complex32_t));
    dim3 threads(256,1,1);
    int  t1 = npoints / threads.x + 1;
    dim3 blocks(t1,1,1);
    makeComplex<<< blocks, threads >>> (d_real, d_imag, npoints, d_cplx);
    

    // build
    int nd = 2;
    int gdims[] = {512, 1};
    af::cfloat * buf = reinterpret_cast<af::cfloat *>(d_cplx);
    PyObject * out =  PyAfnumpy_FromData(nd, gdims, CMPLX32, buf, true);
    return out;
}
