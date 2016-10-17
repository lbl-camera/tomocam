#include <stdio.h>
#include <iostream>
#include <Python.h>
#include <arrayfire.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <numpy/arrayobject.h>

#include "afnumpyapi.h"
#include "pyGnufft.h"

DataType PyAfnumpy_Type(PyObject * in){
    if (PyObject_HasAttrString(in, "d_array")){
        PyObject * d_array = PyObject_GetAttrString(in, "d_array");
        PyObject * type = PyObject_CallMethod(d_array, (char *) "type", NULL);
        Py_DECREF(d_array);
        return (DataType) PyLong_AsLong(type);
    }
    return (DataType) 0;
}

int PyAfnumpy_Size(PyObject * in){
    if (PyObject_HasAttrString(in, "size")){
        PyObject * size = PyObject_GetAttrString(in, "size");
        return PyLong_AsLong(size); 
    }
    return -1;
}

int PyAfnumpy_NumOfDims(PyObject * in){
    if (PyObject_HasAttrString(in, "shape")){
        PyObject * shape = PyObject_GetAttrString(in, "shape");
        if (PyTuple_Check(shape)){
            Py_ssize_t len = PyTuple_Size(shape);
            return (size_t) len;
        }
        return -1;
    }
    return -1;
}

int PyAfnumpy_Dims(PyObject * in, int i){
    if (PyObject_HasAttrString(in, "shape")){
        PyObject * shape = PyObject_GetAttrString(in, "shape");
        if (PyTuple_Check(shape)){
            Py_ssize_t len = PyTuple_Size(shape);
            if ((i < 0) || (i >= len)){
                fprintf(stderr, "error: out of range dimension\n");
                Py_DECREF(shape);
                return -1;
            }
            PyObject * item = PyTuple_GetItem(shape, (Py_ssize_t) i);
            Py_DECREF(shape);
            return PyLong_AsLong(item);
        }
    }
    return -1;
}

void * PyAfnumpy_DevicePtr(PyObject * in){
    if (PyObject_HasAttrString(in, "d_array")){
        // get shape, dims and device_ptr
        PyObject * d_array = PyObject_GetAttrString(in, "d_array");
        PyObject * pointer = PyObject_CallMethod(d_array, (char *) "device_ptr", NULL);
        void * buf = NULL;
        if (PyLong_Check(pointer) || PyInt_Check(pointer))
            buf = PyLong_AsVoidPtr(pointer);
        else {
            PyObject * attrval = PyObject_GetAttrString(pointer, "value");
            if (PyLong_Check(attrval) || PyInt_Check(attrval)) 
                buf = PyLong_AsVoidPtr(attrval);
            else
               fprintf(stderr, "Error: failed to read the device pointer.\n");
            Py_DECREF(attrval); 
        }
        if (!buf) {
            // try the host memory
            fprintf(stderr, "Warning: device_ptr is NULL. Now copying data to device.\n");
            fprintf(stderr, "Error: failed to get data from afnumpy array\n");
            Py_DECREF(d_array);
            Py_DECREF(pointer);
            return NULL;
        } 
        Py_DECREF(d_array);
        Py_DECREF(pointer);
        return buf;
    }
    return NULL;
}

PyObject * PyArrayfire_FromData(int dims_size, int * dims,
            DataType type, void * buffer, bool is_device){

    /* Build arguments for arrayfire.Array() */
    // array dimensions
    PyObject * af_dims =  PyTuple_New(dims_size);
    for (int i = 0; i < dims_size; i++)
        PyTuple_SetItem(af_dims, i, PyLong_FromLong(dims[i]));

    // array af_type
    PyObject * af_type = Py_BuildValue("s", "f");
    switch (type){
        case FLOAT32:
            break;
        case CMPLX32:
            af_type = Py_BuildValue("s", "F");
            break;
        case FLOAT64:
            af_type = Py_BuildValue("s", "d");
            break;
        case CMPLX64: 
            af_type = Py_BuildValue("s", "D");
            break; 
        case BOOL8:
            af_type = Py_BuildValue("s", "b");
            break;
        case INT32:
            af_type = Py_BuildValue("s", "i");
            break;
        case INT64:
            af_type = Py_BuildValue("s", "l");
            break;
        case UINT8:
            PYAF_NOTIMPLEMENTED;
            return NULL;
        case UINT32:
            af_type = Py_BuildValue("s", "I");
            break;
        case UINT64:
            af_type = Py_BuildValue("s", "L");
            break;
    }

    /* pack args into a tuple 
     * arrayfire.Array(buf, dims, dtype, is_device)
     */ 
    PyObject * args = PyTuple_New(4);
    PyTuple_SetItem(args, 0, PyLong_FromVoidPtr(buffer));
    PyTuple_SetItem(args, 1, af_dims);
    PyTuple_SetItem(args, 2, af_type);
    PyTuple_SetItem(args, 3, Py_True);

    /* import arrayfire and create arrayfire.Array object */
    PyObject * arrayfire = PyString_FromString("arrayfire");
    PyObject * module = PyImport_Import(arrayfire);
    PyObject * af = PyModule_GetDict(module);
    PyObject * af_array = PyDict_GetItemString(af, "Array"); 
    PyObject * out = PyObject_CallObject(af_array, args);

    /* garbage collection */
    Py_DECREF(args);
    Py_DECREF(arrayfire);
    Py_DECREF(module);
    return out;
}

PyObject * PyAfnumpy_FromData(int dims_size, int * dims, DataType type,
            void * buffer, bool is_device){

    /* Build an arryfire object-type before everything */
    //PyObject * d_array = PyArrayfire_FromData(dims_size, dims, type, buffer, is_device);

    PyObject * shape = PyTuple_New(dims_size);
    for (int i = 0; i < dims_size; i++)
        PyTuple_SetItem(shape, i, PyLong_FromLong(dims[i]));

    /* afnumpy.ndarray( ) */
    PyObject * af_type = Py_BuildValue("s", "float32");
    switch (type){
        case FLOAT32:
            break;
        case CMPLX32:
            af_type = Py_BuildValue("s", "complex64");
            break;
        case FLOAT64:
            af_type = Py_BuildValue("s", "float64");
            break;
        case CMPLX64: 
            af_type = Py_BuildValue("s", "complex128");
            break; 
        case BOOL8:
            af_type = Py_BuildValue("s", "bool8");
            break;
        case INT32:
            af_type = Py_BuildValue("s", "int32");
            break;
        case INT64:
            af_type = Py_BuildValue("s", "int64");
            break;
        case UINT8:
            PYAF_NOTIMPLEMENTED;
            return NULL;
        case UINT32:
            af_type = Py_BuildValue("s", "uint32");
            break;
        case UINT64:
            af_type = Py_BuildValue("s", "uint64");
            break;
    }

    /* build afnumpy from arrayfire array */
    PyObject * module = PyString_FromString("afnumpy");
    PyObject * afnumpy = PyImport_Import(module);
    PyObject * ndarray = NULL;
    if (PyObject_HasAttrString(afnumpy, "ndarray"))
        ndarray = PyObject_GetAttrString(afnumpy, "ndarray");
    else {
        // clean-up and return 
        Py_DECREF(shape);
        Py_DECREF(af_type);
        Py_DECREF(module);
        Py_DECREF(afnumpy);
        fprintf(stderr,"error: could not load module or attribute error.\n");
        return NULL;
    }

    PyObject * args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, shape);

    PyObject * kwargs = PyDict_New();
    PyObject * key = Py_BuildValue("s", "buffer");
    PyObject * val = PyLong_FromVoidPtr(buffer);
    PyDict_SetItem(kwargs, key, val);
    key = Py_BuildValue("s", "buffer_type");
    val = Py_BuildValue("s", "cuda");
    PyDict_SetItem(kwargs, key, val);
     

    PyObject *out = PyObject_Call(ndarray, args, kwargs);

    /*
    PyTuple_SetItem(args, 1, af_type);
    PyTuple_SetItem(args, 2, Py_None);
    PyTuple_SetItem(args, 3, PyLong_FromLong(0));
    PyTuple_SetItem(args, 4, Py_None);
    PyTuple_SetItem(args, 5, Py_None);
    PyTuple_SetItem(args, 6, d_array);
    PyObject * out = PyObject_CallObject(af_array, args);
    */

    // relese new references
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(module);
    Py_DECREF(afnumpy);
    Py_DECREF(ndarray);
    if (!out) {
        fprintf(stderr, "error: failed to create afnumpy array");
        return NULL;
    }
    return out;
}


