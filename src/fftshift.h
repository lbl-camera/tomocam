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
    DeviceArray<T> fftshift(const DeviceArray<T> &arr, int axis) {

        // calculate shift
        auto dims = arr.dims();
        int3 shift = {0, 0, 0};

        auto calc_shift = [](int N) { return (N % 2 == 0) ? N / 2 : (N + 1) / 2; };

        if (axis & 4) shift.x = calc_shift(dims.x);
        if (axis & 2) shift.y = calc_shift(dims.y);
        if (axis & 1) shift.z = calc_shift(dims.z);
        return gpu::roll(arr, shift);
    }

    template <typename T>
    DeviceArray<T> ifftshift(const DeviceArray<T> &arr, int axis) {

        // calculate shift
        auto dims = arr.dims();
        int3 shift = {0, 0, 0};

        if (axis & 4) shift.x = dims.x / 2;
        if (axis & 2) shift.y = dims.y / 2;
        if (axis & 1) shift.z = dims.z / 2;
        return gpu::roll(arr, shift);
    }
} // namespace tomocam

#endif // FFTSHIFT__H
