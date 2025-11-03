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

namespace tomocam {
    template <typename T>
    DeviceArray<T> project(const DeviceArray<T> &input,
        const NUFFT::Grid<T> &grid) {

        // cast to complex
        auto in2 = complex(input);

        // nufft type 2
        auto out = nufft2d2(in2, grid);

        //  1d inverse fft along columns
        out = gpu::ifftshift(out);
        out = ifft1D(out);
        out = gpu::fftshift(out);

        T scale = static_cast<T>(input.ncols() * input.ncols());
        // cast to real
        return (real(out) / scale);
    }

    // explicit instantiation
    template DeviceArray<float> project<float>(const DeviceArray<float> &,
        const NUFFT::Grid<float> &); 
    template DeviceArray<double> project<double>(const DeviceArray<double> &,
        const NUFFT::Grid<double> &); 

} // namespace tomocam
