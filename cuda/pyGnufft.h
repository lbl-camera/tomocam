
#ifndef PY_GNUFFT__H
#define PY_GNUFFT__H

#include <Python.h>

//PyObject * cPolarBin (PyObject *, PyObject *);
PyObject * cPolarSample (PyObject *, PyObject *);
PyObject * cPolarSampleTranspose (PyObject *, PyObject *);

#if DEBUG
PyObject * cDebug(PyObject *, PyObject *);
#endif

#endif // PY_GNUFFT__H
