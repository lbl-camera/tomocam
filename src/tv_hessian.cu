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

#include "utils.cuh"
#include "potential_function.cuh"

namespace tomocam {

    __global__ void hessian_zero_kernel(dev_arrayf hessian, float sigma) {

        int3 idx = Index3D();

        if (idx < hessian.dims()) {
            float temp = 0.f;
            for (int ix = 0; ix < 3; ix++)
                for (int iy = 0; iy < 3; iy++)
                    for (int iz = 0; iz < 3; iz++)
                        temp += weight(ix, iy, iz) * d2_pot_func_zero(sigma);
            hessian[idx] += temp;
        }
    }

    void add_tv_hessian(DArray<float> &grad, float sigma) {

        // there is only one slice so we dont need multi-gpu machinary
        // move data to gpu
        auto dev_g = DeviceArray_fromHost<float>(grad.dims(), grad.data(), 0);
        Grid grid(dev_g.dims());
        hessian_zero_kernel<<<grid.blocks(), grid.threads()>>>(dev_g, sigma);
    }
} // namespace tomocam
