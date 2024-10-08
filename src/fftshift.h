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

#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "gpu/fftshift.cuh"

#ifndef FFTSHIFT__H
#define FFTSHIFT__H

namespace tomocam {

    template <typename T>
    DeviceArray<T> fftshift(const DeviceArray<T> &arr) {

        // calculate shift
        auto dims = arr.dims();
        int delta = dims.z / 2;
        if (dims.z % 2) delta = (dims.z + 1) / 2;
        return gpu::roll(arr, delta);
    }

    template <typename T>
    DeviceArray<T> ifftshift(const DeviceArray<T> &arr) {

        // calculate shift
        auto dims = arr.dims();
        int delta = dims.z / 2;
        return gpu::roll(arr, delta);
    }

    template <typename T>
    DeviceArray<gpu::complex_t<T>> phase_shift(
        const DeviceArray<gpu::complex_t<T>> &arr, T delta) {

        return gpu::phase_shift(arr, delta);
    }
} // namespace tomocam

#endif // FFTSHIFT__H
