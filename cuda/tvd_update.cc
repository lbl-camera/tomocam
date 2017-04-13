#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <Python.h>

#include "pyGnufft.h"
#include "afnumpyapi.h"
#include "polarsample.h"

PyObject * cTVDUpdate(PyObject *self, PyObject *prhs) {

    PyObject *pyVolume, *pyObjFunc;
    float mrf_p, mrf_sigma;
    if (!(PyArg_ParseTuple(prhs, "ffOO", 
                    &mrf_p,
                    &mrf_sigma,
                    &pyVolume,
                    &pyObjFunc))){
        return NULL;
    }

    // data POINTERS
    complex_t * volume = (complex_t *) PyAfnumpy_DevicePtr(pyVolume);
    complex_t * objfcn = (complex_t *) PyAfnumpy_DevicePtr(pyObjFunc);

    int n1 = PyAfnumpy_Dims(pyVolume, 0);
    int n2 = PyAfnumpy_Dims(pyVolume, 1);
    int n3 = PyAfnumpy_Dims(pyVolume, 2);

    // calculate TVD and add it to objective function
    addTVD(n1, n2, n3, mrf_p, mrf_sigma, objfcn, volume);
    Py_RETURN_NONE;
}



// calculated hessian
PyObject * cCalcHessian(PyObject *self, PyObject *prhs) {
    PyObject * pyVolume;
    float mrf_sigma;
    if (!(PyArg_ParseTuple(prhs, "fO", 
                    &mrf_sigma, 
                    &pyVolume))){
        return NULL;
    }

    // get data
    complex_t * volume = (complex_t *) PyAfnumpy_DevicePtr(pyVolume);
    int n1 = PyAfnumpy_Dims(pyVolume, 0);
    int n2 = PyAfnumpy_Dims(pyVolume, 1);
    int n3 = PyAfnumpy_Dims(pyVolume, 2);

    // allocate memory for hessian
    complex_t * hessian;
    cudaMalloc((void **) &hessian, n1 * n2 * n3 * sizeof(complex_t));

    // compute on GPU
    calcHessian(n1, n2, n3, mrf_sigma, volume, hessian);
   
    // build return afnumpy array 
    int nd = 3;
    int dims[3] = {n1, n2, n3};
    PyObject * out = PyAfnumpy_FromData(nd, dims, CMPLX32, hessian, true);
    return out;
}
