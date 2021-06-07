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

#include "dist_array.h"
#include "machine.h"
#include "optimize.h"
#include "tomocam.h"
#include "types.h"

template <typename T>
inline T *getPtr(np_array_t<T> array) {
    return (T *)array.request().ptr;
}

np_array_t<float> radon_wrapper(np_array_t<float> &imgstack,
    np_array_t<float> angs,
    double cen,
    double oversamp) {
    tomocam::DArray<float> arg1(imgstack);
    int num_angs = static_cast<int>(angs.size());
    tomocam::dim3_t dims = arg1.dims();

    // create output array
    auto sino = py::array_t<float>({dims.x, num_angs, dims.z});
    tomocam::DArray<float> arg2(sino);

    // radon call
    tomocam::radon(
        arg1, arg2, getPtr<float>(angs), (float)cen, (float)oversamp);

    // return numpy array
    return sino;
}

np_array_t<float> iradon_wrapper(np_array_t<float> &sino,
    np_array_t<float> angs,
    double cen,
    double oversamp) {
    tomocam::DArray<float> arg1(sino);
    tomocam::dim3_t dims = arg1.dims();

    // create output array
    auto recon = py::array_t<float>({dims.x, dims.z, dims.z});
    tomocam::DArray<float> arg2(recon);

    // iradon call
    tomocam::iradon(
        arg1, arg2, getPtr<float>(angs), (float)cen, (float)oversamp);

    // return recon as numpy array
    return recon;
}

void gradient_wrapper(np_array_t<float> &recon,
    np_array_t<float> &sino, np_array_t<float> angs,
    double cen, double oversamp) {

    tomocam::DArray<float> arg1(recon);
    tomocam::DArray<float> arg2(sino);
    tomocam::gradient(
        arg1, arg2, getPtr<float>(angs), (float)cen, (float)oversamp);
}

void tv_wrapper(np_array_t<float> &recon,
    np_array_t<float> &gradients,
    double p,
    double s) {
    tomocam::DArray<float> arg1(recon);
    tomocam::DArray<float> arg2(gradients);
    tomocam::add_total_var(arg1, arg2, (float)p, (float)s);
}

np_array_t<float> mbir_wrapper(np_array_t<float> &np_sino,
    np_array_t<float> &np_angles,
    float center,
    int num_iters,
    float oversample,
    float sigma,
    float p) {

    // create DArray from numpy
    tomocam::DArray<float> sino(np_sino);
    tomocam::dim3_t dims = sino.dims();

    // allocate return array
    auto recn = py::array_t<float>({dims.x, dims.z, dims.z});
    tomocam::DArray<float> model(recn);

    // get data pointer to angles
    float *angles = static_cast<float *>(np_angles.request().ptr);
    tomocam::mbir(sino, model, angles, center, num_iters, oversample, sigma, p);

    // return numpy array
    return recn;
}

/* setup methods table */
PYBIND11_MODULE(cTomocam, m) {
    m.doc() = "Python interface to multi-GPU tomocam";

    // DArray class
    py::class_<tomocam::DArray<float>>(m, "DArray")
        .def(py::init<np_array_t<float> &>())
        .def("init",
            static_cast<void (tomocam::DArray<float>::*)(float)>(
                &tomocam::DArray<float>::init),
            "initialize array with a value")
        .def("__add__", 
            [] (tomocam::DArray<float> & a, 
                tomocam::DArray<float> & b) {
                return (a + b); }, 
                py::is_operator())
        .def("__iadd__", 
            [] (tomocam::DArray<float> & a,
                tomocam::DArray<float> & b) {
                a += b;
                return a; },
                py::is_operator())
        .def("__sub__", 
            [] (tomocam::DArray<float> & a,
                tomocam::DArray<float> & b) {
                return (a - b); },
                py::is_operator())
        .def("norm", &tomocam::DArray<float>::norm)
        .def("sum", &tomocam::DArray<float>::sum)
        .def("max", &tomocam::DArray<float>::max)
        .def("min", &tomocam::DArray<float>::min);


    // set gpu paramters
    m.def("set_num_of_gpus", [](int num) {
        tomocam::MachineConfig::getInstance().setNumOfGPUs(num);
    });

    m.def("set_num_of_streams", [](int num) {
        tomocam::MachineConfig::getInstance().setStreamsPerGPU(num);
    });

    m.def("set_slices_per_stream", [](int num) {
        tomocam::MachineConfig::getInstance().setSlicesPerStream(num);
    });

    // radon transform
    m.def("radon", &radon_wrapper);

    // iradon transform
    m.def("iradon", &iradon_wrapper);

    // gradients
    m.def("gradients", &gradient_wrapper);

    // add_tv
    m.def("total_variation", &tv_wrapper);

    // mbir
    m.def("mbir", &mbir_wrapper, "Model-based iterative reconstruction");
}
