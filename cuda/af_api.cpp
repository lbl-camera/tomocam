#include <Python.h>

#include <stdio.h>
#include <iostream>
#include <arrayfire.h>
#include <cuda.h>

#include "af_api.h"
#include "pyGnufft.h"

int PyAF_Size(PyObject *) { return 0; }
int PyAF_NumOfDims(PyObject *) { return 0; }
int PyAF_Dims(PyObject *, int ) { return 0; }
DataType PyAF_Type(PyObject *) { return BOOL8; };
void * PyAF_DevicePtr (PyObject *) { return nullptr; }

af::array PyAF_GetArray(PyObject * obj){
    if (!PyObject_HasAttrString(obj, "arr")){
        std::cerr << "error: python object is not arrayfire array" << std::endl;
        std::exit(1);
    } 
    af::array tmp;
    PyObject * arr = PyObject_GetAttrString(obj, "arr");
    PyObject * ptr = PyObject_GetAttrString(arr, "value");

    af_array ref = (af_array)(PyLong_AsVoidPtr(ptr));
    if (ref) {
        tmp.set(ref);
    }
    return tmp;
}

PyObject * PyAF_FromData(int ndims, int * dims, DataType dtype, void *buf){
    /* import arrayfire and create arrayfire.Array object */
    PyObject * af = PyImport_Import(PyString_FromString("arrayfire"));
    if (!PyObject_HasAttrString(af, "Array")){
        std::cerr << "error: can't find Array in arrayfire" << std::endl;
        std::exit(1);
    }
    PyObject * Array = PyObject_GetAttrString(af, "Array");

    /* build dictionary of keyword-arguments 
     * src: pointer to data
     * dims: list of dimensions
     * dtype: PyUnicode
     * is_device: True
     */
    PyObject * args = PyTuple_New(4);
    // src
    PyObject * ptr  = PyLong_FromVoidPtr(buf);
    PyTuple_SetItem(args, 0, ptr);

    // dims
    PyObject * shape = PyList_New(ndims);
    for (int i = 0; i < ndims; i++) PyList_SetItem(shape, i, PyLong_FromLong(dims[i]));
    PyTuple_SetItem(args, 1, shape);
    

    // dtype 
    PyObject * type = NULL;
    switch (dtype){
        case FLOAT32:
            type = PyString_FromString("f");
            break;
        case CMPLX32:
            type = PyString_FromString("F");
            break;
        case FLOAT64:
            type = PyString_FromString("d");
            break;
        case CMPLX64:
            type = PyString_FromString("D");
            break;
        case BOOL8:
            type = PyString_FromString("b");
            break;
        case INT32:
            type = PyString_FromString("i");
            break;
        case UINT32: 
            type = PyString_FromString("I");
            break;
        default:
            PYAF_NOTIMPLEMENTED;
            std::exit(1);
    } 
    PyTuple_SetItem(args, 2, type);

    // set device to true
    PyTuple_SetItem(args, 3, Py_True);
    PyObject * out = PyObject_CallObject(Array, args);
    if (out != NULL){
        return out;
   }  else {
        Py_XDECREF(out);
        return NULL;
    }
}

PyObject * PyAF_FromArray(af::array & array){
    /* import arrayfire and create arrayfire.Array object */
    PyObject * af = PyImport_ImportModule("arrayfire");
    PyObject * dict = PyModule_GetDict(af);
    PyObject * object = PyDict_GetItemString(dict, "Array"); 
    PyObject * out = PyObject_CallObject(object, NULL);

    /* use ctype c_void_p to set Array.arr to af_array */
    PyObject * ct = PyImport_ImportModule("ctypes");
    PyObject * dict1 = PyModule_GetDict(ct);
    PyObject * c_void_p = PyDict_GetItemString(dict1, "c_void_p");
    PyObject * ptr = PyLong_FromVoidPtr(array.get());
    PyObject * arr = PyObject_CallFunctionObjArgs(c_void_p, ptr, NULL);

    if (PyObject_SetAttrString(out, "arr", arr) == 0){
        return out;
    } else {
        Py_XDECREF(out);
        return NULL;
    }
}
