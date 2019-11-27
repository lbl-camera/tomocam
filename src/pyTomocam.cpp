/* -------------------------------------------------------------------------------
* Tomocam Copyright (c) 2018
*
* The Regents of the University of California, through Lawrence Berkeley National
* Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
*  Energy). All rights reserved.
*
* If you have questions about your rights to use or distribute this software,
* please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
*
* NOTICE. This Software was developed under funding from the U.S. Department of
* Energy and the U.S. Government consequently retains certain rights. As such, the
* U.S. Government has been granted for itself and others acting on its behalf a
* paid-up, nonexclusive, irrevocable, worldwide license in the Software to
* reproduce, distribute copies to the public, prepare derivative works, and
* perform publicly and display publicly, and to permit other to do so.
*---------------------------------------------------------------------------------
*/

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "dist_array.h"

/* setup methods table */
PYBIND11_MODULE(cuTomocam, m) {
    m.doc() = "Python interface to multi-GPU tomocam";

    // DArray class
    m.class_<tomocam::DArray<float>>(m, "DArrayf")
        .def(py::init<int, int, int>());

    // machine config
    m.def("fft1d",  &fft1d,  "One-dimensional FFT of DArray type");
    m.def("ifft1d", &ifft1d, "One-dimensional IFFT of DArray type");
    m.def("fft2d",  &fft2d,  "Two-dimensional FFT of DArray type");
    m.def("ifft2d", &ifft2d, "Two-dimensional IFFT of DArray type");
}
