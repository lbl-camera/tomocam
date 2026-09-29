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

#ifndef SHRINKAGE__CUH
#define SHRINKAGE__CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"

namespace tomocam {
    namespace gpu {

        /**
         * @brief isotropic TV (vector) shrinkage + Bregman update, fused,
         * per-voxel:
         *   sk    = sqrt(sum_i (d{x,y,z} + b{x,y,z})^2)
         *   dc_i  = max(0, sk - lambda_mu) * (d_i + b_i) / (sk + epsilon)
         *   b_i  += d_i - dc_i                      (written in place)
         *
         * @param dx,dy,dz gradient components (grad_u(x)), read-only
         * @param bx,by,bz Bregman accumulator, updated in place
         * @param dcx,dcy,dcz new auxiliary TV variable d, pure output
         * @param lambda_mu params.lambda / params.mu
         * @param epsilon small constant to avoid division by zero
         */
        template <typename T>
        void shrinkage(const DeviceArray<T> &dx, const DeviceArray<T> &dy,
            const DeviceArray<T> &dz, DeviceArray<T> &bx, DeviceArray<T> &by,
            DeviceArray<T> &bz, DeviceArray<T> &dcx, DeviceArray<T> &dcy,
            DeviceArray<T> &dcz, T lambda_mu, T epsilon);

    } // namespace gpu
} // namespace tomocam
#endif // SHRINKAGE__CUH
