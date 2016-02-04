#include "pyGnufft.h"

/* setup methdods table */
static PyMethodDef cGnufftMehods [] = {
	{ "polarbin", polarbin, METH_VARARGS },
    { "polarsample", polarsample, METH_VARARGS },
	{ "polargrid", polargrid, METH_VARARGS },
	{ NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initcGnufft() {
	(void) Py_InitModule("cGnufft", cGnufftMehods);
}

