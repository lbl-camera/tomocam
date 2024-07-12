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

#include "dev_array.h"
#include "fft.h"
#include "fftshift.h"
#include "internals.h"
#include "nufft.h"
#include "types.h"

#include "gpu/padding.cuh"

// #include "debug.h"

namespace tomocam {

    template <typename T>
    DeviceArray<T> backproject(const DeviceArray<T> &sino,
        const NUFFT::Grid<T> &grid, int offset, cudaStream_t s) {

        // add zeropadding to put center of rotation at center of image
        PadType pad_type = PadType::RIGHT;
        if (offset < 0) pad_type = PadType::LEFT;
        auto in1 = gpu::pad1d(sino, 2 * offset, pad_type, s);

        // cast to complex
        auto in2 = complex(in1, s);

        /* back-project */
        // shift 0-frequency to corner
        auto in3 = ifftshift(in2, s);
        auto in4 = fft1D(in3, s);
        auto in5 = in4.divide(in4.ncols(), s);
        auto fk = fftshift(in5, s);

        // wait for fft to finish
        cudaStreamSynchronize(s);

        // allocate memory for back-projection
        dim3_t dims = {fk.nslices(), fk.ncols(), fk.ncols()};
        DeviceArray<gpu::complex_t<T>> out(dims);

        // nufft type 1
        nufft2d1(fk, out, grid);
        cudaDeviceSynchronize();

        // cast to real
        return gpu::unpad2d(real(out, s), offset, s);
    }

    // explicit instantiation
    template DeviceArray<float> backproject(const DeviceArray<float> &,
        NUFFT::Grid<float> const &, int, cudaStream_t);
    template DeviceArray<double> backproject(const DeviceArray<double> &,
        NUFFT::Grid<double> const &, int, cudaStream_t);
} // namespace tomocam
