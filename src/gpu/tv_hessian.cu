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

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "dist_array.h"
#include "internals.h"
#include "types.h"

#include "gpu/utils.cuh"
#include "potential_function.cuh"

namespace tomocam {
    namespace gpu {

        template <typename T>
        __global__ void hessian_zero_kernel(DeviceMemory<T> hessian,
            float sigma) {

            int3 idx = Index3D();

            // clang-format off
            if (idx < hessian.dims()) {
                T temp = 0.f;
                for (int ix = 0; ix < 3; ix++)
                    for (int iy = 0; iy < 3; iy++)
                        for (int iz = 0; iz < 3; iz++)
                            temp += weight(ix, iy, iz) * d2_pot_func_zero(sigma);
                hessian[idx] += temp;
                // clang-format on
            }
        }

        template <typename T>
        void add_tv_hessian(DArray<T> &g, float sigma) {

            // there is only one slice so we dont need multi-gpu machinary
            // move data to gpu
            DeviceArray<T> dev_g(g.dims());
            SAFE_CALL(cudaMemcpy(dev_g.dev_ptr(), g.begin(), g.bytes(),
                cudaMemcpyHostToDevice));

            Grid grid(g.dims());
            hessian_zero_kernel<T>
                <<<grid.blocks(), grid.threads()>>>(dev_g, sigma);
        }

        // instantiate template
        template void add_tv_hessian(DArray<float> &g, float sigma);
        template void add_tv_hessian(DArray<double> &g, float sigma);

    } // namespace gpu
} // namespace tomocam
