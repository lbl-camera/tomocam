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
#include <cuda.h>

#include "pyGnufft.h"
#include "af_api.h"
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
    complex_t * volume = (complex_t *) PyAF_DevicePtr(pyVolume);
    complex_t * objfcn = (complex_t *) PyAF_DevicePtr(pyObjFunc);

    int n1 = PyAF_Dims(pyVolume, 0);
    int n2 = PyAF_Dims(pyVolume, 1);
    int n3 = PyAF_Dims(pyVolume, 2);

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
    complex_t * volume = (complex_t *) PyAF_DevicePtr(pyVolume);
    complex_t * hessian = (complex_t *) PyAF_DevicePtr(pyFcn);
    int n1 = PyAF_Dims(pyVolume, 0);
    int n2 = PyAF_Dims(pyVolume, 1);
    int n3 = PyAF_Dims(pyVolume, 2);

    // compute on GPU
    calcHessian(n1, n2, n3, mrf_sigma, volume, hessian);
    Py_RETURN_NONE;
}
