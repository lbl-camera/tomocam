#include <iostream>

/* setup methdods table */
static PyMethodDef cGnufftMehods [] = {
	{ "polargrid", polargrid, METH_VARARGS },
	{ "polarbin", polarbin, METH_VARARGS },
	{ NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initcGnufft() {
	(void) Py_InitModule("cGnufft", cGnufftMehods);
}

static PyObject * polarbin (PyObject *self, PyObject *args){

}

static PyObject * polargrid (PyObject *self, PyObject *args){


}
