
#ifndef PY_GNUFFT__H
#define PY_GNUFFT__H

#include <Python.h>

PyObject * cPolarSample (PyObject *, PyObject *);
PyObject * cPolarSampleTranspose (PyObject *, PyObject *);
PyObject * cTVDUpdate (PyObject *, PyObject *);
PyObject * cHessian(PyObject *, PyObject *);

PyObject * cDebug(PyObject *, PyObject *);

#endif // PY_GNUFFT__H
