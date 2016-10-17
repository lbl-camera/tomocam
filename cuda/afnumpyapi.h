#ifndef PYAFNUMPYAPI__H
#define PYAFNUMPYAPI__H

#include <Python.h>

/* Macros */
#define PYAF_NOTIMPLEMENTED (fprintf(stderr,"This feature hasn't been implemented." \
        "Please submit a feature request.\n"))

enum DataType { 
    FLOAT32, CMPLX32, FLOAT64, CMPLX64, BOOL8, INT32, UINT32, UINT8, INT64, UINT64
};

int         PyAfnumpy_Size(PyObject *);
int         PyAfnumpy_NumOfDims(PyObject *);
int         PyAfnumpy_Dims(PyObject *, int );
DataType    PyAfnumpy_Type(PyObject *);
void *      PyAfnumpy_DevicePtr (PyObject * in);

PyObject * PyArrayfire_FromData(int, int *, DataType, void * , bool);
PyObject * PyAfnumpy_FromData(int, int *, DataType, void *, bool);

#endif // PYAFNUMPYAPI__H
