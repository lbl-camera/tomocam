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
#include "kernel.h"
#include "types.h"
#include "utils.cuh"

namespace tomocam {

    __global__ 
    void deapodize2d_kernel(dev_arrayc arr, kernel_t kernel) {
        int3 idx = Index3D();
        dim3_t dims = arr.dims();
        float cen_y  = (float) (dims.y / 2);
        float cen_z  = (float) (dims.z / 2);
        if (idx < arr.dims()) {
            float y = (idx.y - cen_y) / dims.y;
            float z = (idx.z - cen_z) / dims.z;
            float wy = kernel.weightft(y);
            float wz = kernel.weightft(z);
            arr[idx].x = arr[idx].x / wz / wy;
        }
    }

    __global__ 
    void deapodize1d_kernel(dev_arrayc arr, kernel_t kernel) {
        int3 idx = Index3D();
        dim3_t dims = arr.dims();
        float cen  = (float) (dims.z / 2);
        if ( idx < dims) {
            float z = (idx.z - cen) /  dims.z;
            float w  = kernel.weightft(z);
            arr[idx].x = arr[idx].x / w;
        }
    }

    // kernel wrappers 
    void deapodize2D(dev_arrayc arr, kernel_t kernel, cudaStream_t stream) {
        Grid grid(arr.dims());
        deapodize2d_kernel <<< grid.blocks(), grid.threads(), 0, stream >>> (arr, kernel);
    }

    void deapodize1D(dev_arrayc arr, kernel_t kernel, cudaStream_t stream) {
        Grid grid(arr.dims());
        deapodize1d_kernel <<< grid.blocks(), grid.threads(), 0, stream >>> (arr, kernel);
    }
} // namespace tomocam
