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
PyObject * cHessian(PyObject *self, PyObject *prhs) {
    PyObject * pyVolume;
    PyObject * pyFcn;
    float mrf_sigma;
    if (!(PyArg_ParseTuple(prhs, "fOO", 
                    &mrf_sigma, 
                    &pyVolume,
                    &pyFcn))){
        return NULL;
    }

    // get data
    complex_t * volume = (complex_t *) PyAfnumpy_DevicePtr(pyVolume);
    complex_t * hessian = (complex_t *) PyAfnumpy_DevicePtr(pyFcn);
    int n1 = PyAfnumpy_Dims(pyVolume, 0);
    int n2 = PyAfnumpy_Dims(pyVolume, 1);
    int n3 = PyAfnumpy_Dims(pyVolume, 2);

    // compute on GPU
    calcHessian(n1, n2, n3, mrf_sigma, volume, hessian);
    Py_RETURN_NONE;
}
