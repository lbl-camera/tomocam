
#ifndef PY_GNUFFT__H
#define PY_GNUFFT__H

#include <Python.h>
#include <arrayfire.h>
/* Macros */
#define PYAF_NOTIMPLEMENTED (fprintf(stderr,"This feature hasn't been implemented." \
        "Please submit a feature request.\n"))

enum DataType { 
    FLOAT32, CMPLX32, FLOAT64, CMPLX64, BOOL8, INT32, UINT32, UINT8, INT64, UINT64
};

af::array PyAfnumpy_AsArrayfireArray(PyObject * in, DataType type);
PyObject * PyArrafire_Array(int, int *, DataType, void * , bool);
PyObject * PyAfnumpy_Array(int, int *, DataType, void *, bool);

static PyObject * polarbin(PyObject *, PyObject *);
static PyObject * polarsample(PyObject *, PyObject *);
static PyObject * polargrid(PyObject *, PyObject *);


#endif // PY_GNUFFT__H
