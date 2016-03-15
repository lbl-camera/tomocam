#include <stdio.h>
#include <Python.h>
#include <arrayfire.h>
#include "pyGnufft.h"
#include <numpy/arrayobject.h>

af::array * PyAfnumpy_AsArrayfireArray(PyObject * in, DataType type){
    if (PyObject_HasAttrString(in, "d_array")){
        void * buf = NULL;
        void * h_buf = NULL;
        bool is_device = true;
        // get shape, dims and device_ptr
        PyObject * af_array = PyObject_GetAttrString(in, "d_array");
        PyObject * device_ptr_call = Py_BuildValue("s", "device_ptr");
        PyObject * device_ptr = PyObject_CallMethodObjArgs(af_array, device_ptr_call, NULL);
        if (!buf) {
            // try the host memory
            PyObject * ndarray = PyObject_GetAttrString(in, "h_array");
            is_device = false;
            h_buf = PyArray_DATA(ndarray);
            if (!h_buf){
                fprintf(stderr, "Error: failed to get data from afnumpy array\n");
                return NULL;
            }
        } else {
            buf = PyLong_AsVoidPtr(device_ptr);
        }
        PyObject * type_call = Py_BuildValue("s", "type");
        PyObject * af_type = PyObject_CallMethodObjArgs(af_array, type_call, NULL);
        if(type != PyInt_AsLong(af_type)){
            fprintf(stderr,"Error: data mismatch in unpack_af_array.");
            return NULL;
        }

        // get number of elements
        PyObject * shape = PyObject_GetAttrString(in, "shape");
        if (!PyTuple_Check(shape)){
            fprintf(stderr, "Error: non-tuple shape returned\n");
            return NULL;
        }
        //PyObject * shape_len = PyObject_CallMethod(shape,(char *)"__len__",NULL);
        Py_ssize_t shape_len = PyTuple_Size(shape);
        if ((shape_len <= 0) || (shape_len > 4)){
            fprintf(stderr,"Error: illegal dimensions");
            return NULL;
        }
        int * dims = new int[shape_len];
        for (Py_ssize_t i = 0; i < shape_len; i++){
            PyObject * num = PyTuple_GetItem(shape, i);
            dims[i] = PyInt_AsLong(num);
            Py_DECREF(num);
        }
        
        int nd[2];
        nd[0] = dims[0]; 
        if (shape_len == 1){
            nd[1] = 1;
        } else if (shape_len == 2) {
            nd[1] = dims[1];
        } else {
            fprintf(stderr, "Only 1-D and 2-D arrays are supported. For higher dims, submit a feature request\n");
            delete [] dims;
            Py_DECREF(af_array);
            Py_DECREF(device_ptr_call);
            Py_DECREF(type_call);
            Py_DECREF(device_ptr);
            Py_DECREF(af_type);
            Py_DECREF(shape);
            return NULL;
        }
        af::array * out = NULL;
        if (type == FLOAT32) {
            if (is_device)
                out = new  af::array(nd[0], nd[1], (float *) buf, afDevice);
            else 
                out = new  af::array(nd[0], nd[1], (float *) h_buf);
        } else if (type == CMPLX32) {
            if (is_device)
                out = new  af::array(nd[0], nd[1], (af::cfloat *) buf, afDevice);
            else 
                out = new  af::array(nd[0], nd[1], (af::cfloat *) h_buf);
        } else if (type == FLOAT64) {
            if (is_device)
                out = new  af::array(nd[0], nd[1], (double *) buf, afDevice);
            else
                out = new  af::array(nd[0], nd[1], (double *) h_buf);
        } else if (type == CMPLX64) {
            if(is_device)
                out = new  af::array(nd[0], nd[1], (af::cdouble *) buf, afDevice);
            else
                out = new  af::array(nd[0], nd[1], (af::cdouble *) h_buf);
        } else if (type == INT32) {
            if (is_device)
                out = new  af::array(nd[0], nd[1], (int *) buf, afDevice);
            else
                out = new  af::array(nd[0], nd[1], (int *) h_buf);
        } else {
            PYAF_NOTIMPLEMENTED;
            return NULL;
        }
        delete [] dims;
        Py_DECREF(af_array);
        Py_DECREF(device_ptr_call);
        Py_DECREF(type_call);
        Py_DECREF(device_ptr);
        Py_DECREF(af_type);
        Py_DECREF(shape);
<<<<<<< HEAD
=======
        //Py_DECREF(shape_len);
>>>>>>> 3b2b07e5aca446f3457359ab7c370bf2e5187ac0
        return out;
    }
    return NULL;
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


