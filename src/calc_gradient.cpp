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

// cuda headers
#include <cuda_runtime.h>
#include <cuda.h>

// std headers
#include <tuple>

// local headers
#include "dev_array.h"
#include "nufft.h"
#include "toeplitz.h"

namespace tomocam {

    /* gradients using Toeplitz matrix structure */
    template <typename T>
    std::tuple<DeviceArray<T>, T> gradient2(DeviceArray<T> &f,
        DeviceArray<T> &y, const PointSpreadFunction<T> &psf,
        cudaStream_t stream) {

        // convolve f with psf
        auto AtAf = psf.convolve(f, stream);

        // compute gradient
        // \f$ \nabla f = A^T A f - A^T y \f$
        auto grad = AtAf.subtract(y, stream);

        return grad;
    }

    // explicit instantiation
    template std::tuple<DeviceArray<float>, float> gradient2(
        DeviceArray<float> &, DeviceArray<float> &,
        const PointSpreadFunction<float> &, cudaStream_t);
    template std::tuple<DeviceArray<double>, double> gradient2(
        DeviceArray<double> &, DeviceArray<double> &,
        const PointSpreadFunction<double> &, cudaStream_t);
}
