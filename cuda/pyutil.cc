#include <stdio.h>
#include <Python.h>
#include <arrayfire.h>
#include "pyGnufft.h"

af::array PyAfnumpy_AsArrayfireArray(PyObject * in, DataType type){
    if (PyObject_HasAttrString(in, "d_array")){
        // get shape, dims and device_ptr
        PyObject * af_array = PyObject_GetAttrString(in, "d_array");
        PyObject * device_ptr = PyObject_CallMethod(af_array, (char *) "device_ptr", NULL);
        PyObject * af_type = PyObject_CallMethod(af_array, (char *) "type", NULL);
        if(type != PyInt_AsLong(af_type)){
            fprintf(stderr,"Error: data mismatch in unpack_af_array.");
            return NULL;
        }

        // get number of elements
        PyObject * shape = PyObject_GetAttrString(in, "shape");
        PyObject * shape_len = PyObject_CallMethod(shape,(char *)"__len__",NULL);
        int ndims = PyInt_AsLong(shape_len);
        Py_DECREF(shape_len);
        size_t length = 1;
        int * dims = new int[ndims];
        for(int i = 0; i < ndims; i++){
            PyObject * dim_len = PyObject_CallMethod(shape,(char *)"__getitem__", (char *)"i", i);
            dims[i] = PyInt_AsLong(dim_len);
            length *= dims[i];
            Py_DECREF(dim_len);
        }
        void * dev_ptr = PyLong_AsVoidPtr(device_ptr);
        int nd[2];
        nd[0] = dims[0]; 
        if (ndims == 1){
            nd[1] = 0;
        } else if (ndims == 2) {
            nd[1] = dims[1];
        } else {
            fprintf(stderr, "Only 1-D and 2-D arrays are supported. For higher dims, submit a feature request\n");
            delete [] dims;
            Py_DECREF(af_array);
            Py_DECREF(device_ptr);
            Py_DECREF(af_type);
            Py_DECREF(shape);
            Py_DECREF(shape_len);
            return NULL;
        }
        af::array out(nd[0], nd[1], dev_ptr, afDevice);
        delete [] dims;
        Py_DECREF(af_array);
        Py_DECREF(device_ptr);
        Py_DECREF(af_type);
        Py_DECREF(shape);
        Py_DECREF(shape_len);
        return out;
    }
}


PyObject * PyArrafire_Array(int dims_size, int * dims,
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
    PyObject * pyDev = Py_False;
    if (is_device)
        pyDev = Py_True;
    PyTuple_SetItem(args, 0, PyLong_FromVoidPtr(buffer));
    PyTuple_SetItem(args, 1, af_dims);
    PyTuple_SetItem(args, 2, af_type);
    PyTuple_SetItem(args, 3, pyDev);

    /* import arrayfire and create arrayfire.Array object */
    PyObject * arrayfire = PyString_FromString("arrayfire");
    PyObject * module = PyImport_Import(arrayfire);
    PyObject * af = PyModule_GetDict(module);
    PyObject * af_array = PyDict_GetItemString(af, "Array"); 
    PyObject * out = PyObject_CallObject(af_array, args);

    /* garbage collection */
    Py_DECREF(af_dims);
    Py_DECREF(af_type);
    Py_DECREF(arrayfire);
    Py_DECREF(module);
    Py_DECREF(af);
    Py_DECREF(af_array);
    Py_DECREF(args);
    return out;
}

PyObject * PyAfnumpy_Array(int dims_size, int * dims, DataType type,
            void * buffer, bool is_device){

    /* Build an arryfire object-type before everything */
    PyObject * d_array = PyArrafire_Array(dims_size, dims, type, buffer, is_device);

    PyObject * shape = PyTuple_New(dims_size);
    for (int i = 0; i < dims_size; i++)
        PyTuple_SetItem(shape, i, PyLong_FromLong(dims[i]));

    /* afnumpy.ndarray( ) */
    PyObject * af_type;
    switch (type){
        case FLOAT32:
            af_type = Py_BuildValue("s", "float32");
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
    PyObject * af = PyString_FromString("afnumpy");
    PyObject * afmod = PyImport_Import(af);
    PyObject * afnumpy = PyModule_GetDict(afmod);
    PyObject * af_array = PyDict_GetItemString(afnumpy, "ndarray");


    PyObject * args = PyTuple_New(7);
    PyTuple_SetItem(args, 0, shape);
    PyTuple_SetItem(args, 1, af_type);
    PyTuple_SetItem(args, 2, Py_None);
    PyTuple_SetItem(args, 3, PyLong_FromLong(0));
    PyTuple_SetItem(args, 4, Py_None);
    PyTuple_SetItem(args, 5, Py_None);
    PyTuple_SetItem(args, 6, d_array);
    PyObject * out = PyObject_CallObject(af_array, args);

    /* collect garbage */
    Py_DECREF(shape);
    Py_DECREF(af_type);
    Py_DECREF(d_array);
    Py_DECREF(af);
    Py_DECREF(afmod);
    Py_DECREF(afnumpy);
    Py_DECREF(af_array);
    Py_DECREF(args);
    return out;
}