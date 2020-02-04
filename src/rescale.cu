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
#include "types.h"

namespace tomocam {
    __global__ void rescale_kernel(cuComplex_t * arr, float scale, int len) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < len) {
            arr[i].x *= scale;
            arr[i].y *= scale;
        } 
    }

    void rescale(cuComplex_t * arr, dim3_t dims, float scale, cudaStream_t stream) {
        int len = dims.x * dims.y * dims.z; 
        dim3 threads(256);
        dim3 tblocks(len / threads.x + 1);
        rescale_kernel <<< tblocks, threads, 0, stream >>> (arr, scale, len);
    }
} // namespace tomocam
