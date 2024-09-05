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

#include <concepts>
#include <iostream>
#include <vector>

#include "dev_array.h"
#include "fft.h"
#include "fftshift.h"
#include "internals.h"
#include "nufft.h"
#include "types.h"

#include "gpu/padding.cuh"

namespace tomocam {

    template <typename T>
    DeviceArray<T> backproject(const DeviceArray<T> &sino,
        const NUFFT::Grid<T> &grid, int offset) {

        // add zeropadding to put center of rotation at center of image
        PadType pad_type = PadType::RIGHT;
        if (offset < 0) pad_type = PadType::LEFT;
        auto in1 = gpu::pad1d(sino, 2 * offset, pad_type);


        // cast to complex
        auto in2 = complex(in1);

        /* back-project */
        // shift 0-frequency to corner
        in2 = ifftshift(in2);
        in2 = fft1D(in2);
        in2 = fftshift(in2);

        // nufft type 1
        auto out = nufft2d1(in2, grid);
        cudaDeviceSynchronize();

        // cast to real
        return gpu::unpad2d(real(out), offset);
    }

    // explicit instantiation
    template DeviceArray<float> backproject(const DeviceArray<float> &,
        NUFFT::Grid<float> const &, int);
    template DeviceArray<double> backproject(const DeviceArray<double> &,
        NUFFT::Grid<double> const &, int);
} // namespace tomocam
