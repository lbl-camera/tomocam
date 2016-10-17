
#include <Python.h>
#include <iostream>

#define PY_ARRAY_UNIQUE_SYMBOL gnufft_ARRAY_API
#include <numpy/arrayobject.h>

#include "pyGnufft.h"

static PyObject * polarsample(PyObject * self, PyObject * args){
    PyObject * res = cPolarSample(self, args);
    return res;
}
static PyObject * polarsample_transpose(PyObject * self, PyObject * args){
    PyObject * res = cPolarSampleTranspose(self, args);
    return res;
}

/*
static PyObject * debug(PyObject * self, PyObject * args){
#if DEBUG
    PyObject * res = cDebug(self, args);
#else
    return NULL;
#endif
}
*/

PyDoc_STRVAR(module_doc, "C++/CUDA extnsions for Radon/Iradon transforms.\n");
PyDoc_STRVAR(fwd_doc, "C++/CUDA extension to map cartesian grid to polar grid\n");
PyDoc_STRVAR(rev_doc, "C++/CUDA extension to map polar grid to cartesian grid\n");

/* setup methdods table */
static PyMethodDef cGnufftMehods [] = {
    { "polarsample",           polarsample,           METH_VARARGS, fwd_doc },
	{ "polarsample_transpose", polarsample_transpose, METH_VARARGS, rev_doc },
	{ NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initgnufft() {
	(void) Py_InitModule3("gnufft", cGnufftMehods, module_doc);
    import_array();
}

