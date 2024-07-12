/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions abo ut your rights to use or distribute this software,
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

#include "dev_array.h"
#include "fft.h"
#include "fftshift.h"
#include "gpu/padding.cuh"
#include "internals.h"
#include "nufft.h"
#include "tomocam.h"
#include "types.h"

#include "debug.h"

namespace tomocam {
    template <typename T>
    DeviceArray<T> project(const DeviceArray<T> &input,
        const NUFFT::Grid<T> &grid, int offset, cudaStream_t s) {

        // zero-padding
        auto in1 = gpu::pad2d(input, 2 * offset, PadType::RIGHT, s);

        // cast to complex
        auto in2 = complex(in1, s);

        // allocate nufft output
        DeviceArray<gpu::complex_t<T>> out(
            dim3_t(input.nslices(), grid.nprojs(), grid.npixels()));

        // nufft type 2
        nufft2d2(out, in2, grid);
        SAFE_CALL(cudaDeviceSynchronize());

        // write_h5(out);
        //  1d inverse fft along columns
        auto out1 = fftshift(out, s);
        auto out2 = ifft1D(out1, s);
        write_h5(out2);
        auto out3 = out2.divide(out2.ncols(), s);
        auto out4 = ifftshift(out3, s);

        // cast to real
        auto out5 = real(out4, s);

        // crop and return
        if (offset == 0) {
            return out5;
        } else {
            PadType pad_type = offset < 0 ? PadType::LEFT : PadType::RIGHT;
            return gpu::unpad1d(out5, offset, pad_type, s);
        }
    }

    // explicit instantiation
    template DeviceArray<float> project<float>(const DeviceArray<float> &,
        const NUFFT::Grid<float> &, int, cudaStream_t);
    template DeviceArray<double> project<double>(const DeviceArray<double> &,
        const NUFFT::Grid<double> &, int, cudaStream_t);

} // namespace tomocam
