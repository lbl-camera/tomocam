
#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL gnufft_ARRAY_API
#include <numpy/arrayobject.h>

#include "pyGnufft.h"


/*
static PyObject * polarbin(PyObject * self, PyObject * args){
    PyObject * res = cPolarBin(self, args);
    return res;
}
*/

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

/* setup methdods table */
static PyMethodDef cGnufftMehods [] = {
    { "polarsample", polarsample, METH_VARARGS },
	{ "polarsample_transpose", polarsample_transpose, METH_VARARGS },
	{ NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initgnufft() {
	(void) Py_InitModule("gnufft", cGnufftMehods);
    import_array();
}

