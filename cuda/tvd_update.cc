#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <Python.h>

#include "pyGnufft.h"
#include "afnumpyapi.h"
#include "polarsample.h"

PyObject * cTVDUpdate(PyObject *self, PyObject *prhs) {

    PyObject *pyVolume, *pyObjFunc;
    if (!(PyArg_ParseTuple(prhs, "OO", 
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

    //printf("n1 = %d, n2 = %d, n3 = %d\n", n1, n2, n3);
    // calculate TVD and add it to objective function
    addTVD(n1, n2, n3, objfcn, volume);
    Py_RETURN_NONE;
}
