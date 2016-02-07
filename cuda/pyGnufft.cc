
#include <Python.h>
#include <numpy/arrayobject.h>
#include "pyGnufft.h"


static PyObject * polarbin(PyObject * self, PyObject * args){
    PyObject * res = cPolarBin(self, args);
    return res;
}

static PyObject * polarsample(PyObject * self, PyObject * args){
    PyObject * res = cPolarSample(self, args);
    return res;
}
static PyObject * polargrid(PyObject * self, PyObject * args){
    PyObject * res = cPolarGrid(self, args);
    return res;
}

/* setup methdods table */
static PyMethodDef cGnufftMehods [] = {
	{ "polarbin", polarbin, METH_VARARGS },
    { "polarsample", polarsample, METH_VARARGS },
	{ "polargrid", polargrid, METH_VARARGS },
	{ NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initgnufft() {
	(void) Py_InitModule("gnufft", cGnufftMehods);
    import_array();
}

