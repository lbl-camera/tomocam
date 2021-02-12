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

#include "types.h"
#include "dist_array.h"
#include "tomocam.h"
#include "machine.h"

template <typename T>
inline T * getPtr(np_array_t<T> array) {
    return (T *) array.request().ptr;
}

void radon_wrapper(tomocam::DArray<float> &arg1, tomocam::DArray<float> &arg2, np_array_t<float> angs, double cen, double oversamp) {
    tomocam::radon(arg1, arg2, getPtr<float>(angs), (float) cen, (float) oversamp);
}
        
void iradon_wrapper(tomocam::DArray<float> &arg1, tomocam::DArray<float> &arg2, np_array_t<float> angs, double cen, double oversamp) {
    tomocam::iradon(arg1, arg2, getPtr<float>(angs), (float) cen, (float) oversamp);
}
        
void gradient_wrapper(tomocam::DArray<float> &arg1, tomocam::DArray<float> &arg2, np_array_t<float> angs, double cen, double oversamp) {
    tomocam::gradient(arg1, arg2, getPtr<float>(angs), (float) cen, (float) oversamp);
}
        
void tv_wrapper(tomocam::DArray<float> &arg1, tomocam::DArray<float> &arg2, double p, double s) {
    tomocam::add_total_var(arg1, arg2, (float) p, (float) s);
}
 

/* setup methods table */
PYBIND11_MODULE(cTomocam, m) {
    m.doc() = "Python interface to multi-GPU tomocam";

    // DArray class
    py::class_<tomocam::DArray<float>>(m, "DArray")
        .def(py::init<np_array_t<float> &>())
        .def("init", static_cast<void (tomocam::DArray<float>::*)(float)>(&tomocam::DArray<float>::init), "initialize array with a value");

    // set gpu paramters
    m.def("set_num_of_gpus", 
            [](int num) {
                tomocam::MachineConfig::getInstance().setNumOfGPUs(num);
            }
        );

    m.def("set_num_of_streams",
            [](int num) {
                tomocam::MachineConfig::getInstance().setStreamsPerGPU(num);
            }
    );

    m.def("set_slices_per_stream",
            [](int num) {
                tomocam::MachineConfig::getInstance().setSlicesPerStream(num);
            }
    );
    
    // radon transform
    m.def("radon", &radon_wrapper); 

    // iradon transform
    m.def("iradon", &iradon_wrapper);

    // gradients
    m.def("gradients", &gradient_wrapper);

    // add_tv
    m.def("total_variation", &tv_wrapper);

    // caxpy
    m.def("axpy", &tomocam::axpy);

    // norm2
    m.def("norm", &tomocam::norm2);    
}
