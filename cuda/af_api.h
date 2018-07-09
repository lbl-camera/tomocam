#ifndef PYAFNUMPYAPI__H
#define PYAFNUMPYAPI__H

#include <Python.h>
#include <arrayfire.h>

/* Macros */
#define PYAF_NOTIMPLEMENTED (fprintf(stderr,"This feature hasn't been implemented." \
        "Please submit a feature request.\n"))

enum DataType { 
    FLOAT32, CMPLX32, FLOAT64, CMPLX64, BOOL8, INT32, UINT32, UINT8, INT64, UINT64
};

int PyAF_Size(PyObject *); 
int PyAF_NumOfDims(PyObject *);
int PyAF_Dims(PyObject *, int ); 
DataType PyAF_Type(PyObject *); 
void * PyAF_DevicePtr (PyObject *); 
    //def __init__(self, src=None, dims=None, dtype=None, is_device=False, offset=None, strides=None):

PyObject * PyAF_FromArray(af::array &);
PyObject * PyAF_FromData(int, int *, DataType, void *);
af::array PyAF_GetArray(PyObject *);
#endif // PYAFNUMPYAPI__H
