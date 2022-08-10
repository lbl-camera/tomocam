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

    void rescale(dev_arrayc &arr, float scale, cudaStream_t stream) {
        Grid grid(arr.dims()); 
        rescale_kernel <<< grid.blocks(), grid.threads(), 0, stream >>> (arr, scale);
    }


    __global__ void filter_gradient_kernel(DeviceArray<cuComplex_t> arr) {
        int3 idx = Index3D();
        dim3_t dims = arr.dims();
        if (idx < dims) {
            float x0 = dims.z / 2.;
            float y0 = dims.y / 2.;
            float qx = (idx.z - x0) / idx.z;
            float qy = (idx.y - y0) / idx.y;
            float q = sqrt(qx*qx + qy*qy);
            arr[idx] = arr[idx] * q;
        }
    }

    void filter_gradient(DeviceArray<cuComplex_t> &arr, cudaStream_t stream) {
        Grid grid(arr.dims());
        filter_gradient_kernel <<<grid.blocks(), grid.threads(), 0, stream >>>(arr);
    }
} // namespace tomocam
