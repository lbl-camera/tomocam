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
#include "utils.cuh"

namespace tomocam {
    __global__ void rescale_kernel(DeviceArray<cuComplex_t> arr, float scale) {
        int3 idx = Index3D();
        if (idx < arr.dims()) 
            arr[idx] = arr[idx]*scale;
    }

    void rescale(dev_arrayc arr, cudaStream_t stream) {
        dim3_t dims = arr.dims();
        float scale = 1.f / (dims.z * dims.z);
        Grid grid(dims); 
        rescale_kernel <<< grid.blocks(), grid.threads(), 0, stream >>> (arr, scale);
    }
} // namespace tomocam
