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

#include "dist_array.h"
#include "types.h"

#include "utils.cuh"

namespace tomocam {
    __global__ void calc_error_kernel(cuComplex_t * model, float * data, dim3_t d1, dim3_t d2) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;

        if ((i < d1.x) && (j < d1.y) && (k < d1.z)) {
            int m_id = i * d1.y * d1.z + j * d1.z + k;
            int ipad = (d1.z - d2.z)/2;
            if ((k < ipad) || (k > ipad + d2.z - 1)) {
                model[m_id].x = 0.f;
                model[m_id].y = 0.f;
            } else {
                int d_id = i * d2.y * d2.z + j * d2.z + k - ipad;
                model[m_id].x = data[d_id] - model[m_id].x;
                model[m_id].y = 0.f;
            }
        }
    }

    void calc_error(cuComplex_t *model,  float *data, dim3_t d1, dim3_t d2, cudaStream_t stream) {
        dim3 threads(1, 16, 16);
        dim3 dims(d1.x, d1.y, d1.z);
        dim3 tblocks = calcBlocks(dims, threads);
        calc_error_kernel <<< tblocks, threads, 0, stream >>> (model, data, d1, d2);
    }
} // namespace tomocam
