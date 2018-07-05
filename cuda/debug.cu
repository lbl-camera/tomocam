#include <Python.h>

#include <iostream>
#include <arrayfire.h>
#include <cuComplex.h>

#include <cuda.h>

#include "pyGnufft.h"
#include "af_api.h"

typedef cuComplex complex32_t;


__global__ void makeComplex(float * real, float * imag, int npts, 
                       complex32_t * cplx){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < npts )
        cplx[i] = make_cuComplex(real[i], imag[i]);
}

PyObject *cDebug(PyObject *self, PyObject *prhs) {

    PyObject *in0, *in1;
    if (!(PyArg_ParseTuple(prhs, "OO", &in0, &in1))){
        return NULL;
    }
    af::array real = PyAF_GetArray(in0);
    af::array imag = PyAF_GetArray(in1);

    // data POINTERS
    int nreal = real.elements();
    int nimag = imag.elements();
    if ( nreal != nimag ) {
        std::cerr << "error: input array must of same size" << std::endl;
        exit(1);
    }
    float * d_real = real.device<float>();
    float * d_imag = imag.device<float>();
   
    // allocate memory for complex
    af::array cplx = af::constant(0, real.dims(), c32);
    complex32_t * d_cplx = (complex32_t *) cplx.device<af::cfloat>();

    dim3 threads(256,1,1);
    if (nreal < 256)
        threads.x = nreal;

    int  t1 = nreal / threads.x + 1;
    dim3 blocks(t1,1,1);
    makeComplex<<< blocks, threads >>> (d_real, d_imag, nreal, d_cplx);
   
    // build pyObject
    cplx.unlock();
    real.unlock();
    imag.unlock();

    std::cout << "af_array: " << cplx.get() << std::endl;

    PyObject * res = PyAF_FromArray(cplx);
    if (res != NULL){
        return res;
    } else {
        std::cerr << "error: crapped" << std::endl;
        exit(1);
    }
}
