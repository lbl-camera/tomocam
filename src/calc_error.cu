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
#include "utils.cuh"

namespace tomocam {
    __global__ void calc_error_kernel(dev_arrayc model, dev_arrayf data, int ipad) {

        int3 idx = Index3D();
        dim3_t d1 = model.dims();
        dim3_t d2 = data.dims();
        int n2 = d2.z - 1 + ipad;
        if (idx < d1) {
            int ipad = (d1.z - d2.z) / 2;
            if ((idx.z < ipad) || (idx.z > n2)) {
                model[idx].x = 0.f;
                model[idx].y = 0.f;
            } else {
                model[idx].x = model[idx].x - data(idx.x, idx.y, idx.z - ipad);
                model[idx].y = 0.f;
            }
        }
    }

    void calc_error(dev_arrayc &model, dev_arrayf &data, int ipad, cudaStream_t stream) {
        Grid grid(model.dims());
        calc_error_kernel<<<grid.blocks(), grid.threads(), 0, stream>>>(model, data, ipad);
    }
} // namespace tomocam
