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
#include "gpu/filters.cuh"

#ifdef DEBUG
#include "debug.h"
#endif


namespace tomocam {

    template <typename T>
    DeviceArray<T> backproject(const DeviceArray<T> &sino,
        const NUFFT::Grid<T> &grid, bool fbp) {

        // cast to complex
        auto in2 = complex(sino);

        /* back-project */
        // shift 0-frequency to corner
        in2 = gpu::ifftshift(in2);

        // forward FFT in radial direction
        in2 = fft1D(in2);
        // shift 0-frequency  to center
        in2 = gpu::fftshift(in2);

        if (fbp) gpu::apply_filter(in2);

        // nufft type 1
        auto out = nufft2d1(in2, grid);
        SAFE_CALL(cudaDeviceSynchronize());

        // return real part
        T scale =  static_cast<T>(sino.ncols() * sino.ncols());
        return (real(out) / scale);
    }

    // explicit instantiation
    template DeviceArray<float> backproject(const DeviceArray<float> &,
        NUFFT::Grid<float> const &, bool);
    template DeviceArray<double> backproject(const DeviceArray<double> &,
        NUFFT::Grid<double> const &, bool);
} // namespace tomocam
