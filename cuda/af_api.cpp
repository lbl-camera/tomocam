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
PyObject * PyAF_FromData(int, int *, DataType, void *){ return NULL; }

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
    af_print(tmp);
    return tmp;
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
