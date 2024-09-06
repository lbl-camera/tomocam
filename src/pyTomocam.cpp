/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <iostream>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "dist_array.h"
#include "machine.h"
#include "nufft.h"
#include "optimize.h"
#include "tomocam.h"
#include "types.h"

namespace py = pybind11;

template <typename T>
using np_array_t = py::array_t<T, py::array::c_style>;

template <typename T>
inline T *getPtr(np_array_t<T> array) {
    return (T *)array.request().ptr;
}

template <typename T>
inline std::vector<T> getVec(np_array_t<T> array) {
    auto buffer_info = array.request();
    return std::vector<T>((T *)buffer_info.ptr,
        (T *)buffer_info.ptr + buffer_info.size);
}

template<typename T> 
inline tomocam::DArray<T> from_numpy(const np_array_t<T> &np_arr) {
    auto buffer_info = np_arr.request();
    tomocam::dim3_t dims(1, 1, 1);
    if (buffer_info.ndim == 2) {
        dims.y = np_arr.shape(0);
        dims.z = np_arr.shape(1);
    } else if (buffer_info.ndim == 3) {
        dims.x = np_arr.shape(0);
        dims.y = np_arr.shape(1);
        dims.z = np_arr.shape(2);
    }
    tomocam::DArray<T> rv(dims);
    std::copy((T *)buffer_info.ptr, (T *)buffer_info.ptr + rv.size(),
        rv.begin());
    return rv;
}

template<typename T>
inline np_array_t<T> to_numpy(const tomocam::DArray<T> &arr) {
    auto dims = arr.dims();
    std::vector<ssize_t> shape{(ssize_t) dims.x, (ssize_t)dims.y, (ssize_t)dims.z};
    size_t N = arr.size();
    T * buf = new T [N];
    std::copy(arr.begin(), arr.end(), buf);
    return np_array_t<T>(shape, buf);
}

np_array_t<float> 
radon_wrapper(np_array_t<float> &imgstack, np_array_t<float> angs, int cen) {

    // create DArray from numpy
    tomocam::DArray<float> arg1(from_numpy<float>(imgstack));

    // radon call
    auto arg2 = tomocam::project(arg1, getVec<float>(angs), cen);

    // return numpy array
    return to_numpy<float>(arg2);
}

np_array_t<float> 
backproject_wrapper(np_array_t<float> &sino, np_array_t<float> angs, int cen) {

    // create DArray from numpy
    tomocam::DArray<float> arg1(from_numpy<float>(sino));

    // backproject call
    auto arg2 = tomocam::backproject(arg1, getVec<float>(angs), cen);

    // return recon as numpy array
    return to_numpy<float>(arg2);
}

np_array_t<float> mbir_wrapper(np_array_t<float> &np_sino,
    np_array_t<float> &np_angles, int center, int num_iters, float sigma,
    float tol, float step_size, float penalty) {

    // create DArray from numpy
    tomocam::DArray<float> sino(from_numpy<float>(np_sino));
    tomocam::dim3_t dims = sino.dims();

    // get data pointer to angles
    float p = 1.2;
    auto angles = getVec<float>(np_angles);
    tomocam::DArray<float> recon = 
        tomocam::mbir(sino, angles, center, sigma, p, num_iters, step_size, tol, penalty);

    // return numpy array
    return to_numpy<float>(recon);
}

/* setup methods table */
PYBIND11_MODULE(cTomocam, m) {
    m.doc() = "Python interface to multi-GPU tomocam";

    // set gpu paramters
    m.def("set_num_of_gpus", [](int num) {
        tomocam::Machine::config.setNumOfGPUs(num);
    });

    m.def("set_slices_per_stream", [](int num) {
        tomocam::Machine::config.setSlicesPerStream(num);
    });

    // radon transform
    m.def("radon", &radon_wrapper);

    // radon transform
    m.def("backproject", &backproject_wrapper);

    // mbir
    m.def("mbir", &mbir_wrapper, "Model-based iterative reconstruction");
}
