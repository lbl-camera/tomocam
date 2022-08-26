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
#include "nufft.h"

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


    __global__ void filter_gradient_kernel(DeviceArray<cuComplex_t> arr, float *x, float *y) {
        int3 idx = Index3D();
        dim3_t dims = arr.dims();
        if (idx < dims) {
            int j = idx.y * dims.z + idx.z;
            float w = sqrt(x[j] * x[j] + y[j] * y[j]);
            arr[idx] = arr[idx] * w;
        }
    }

    void filter_gradient(DeviceArray<cuComplex_t> &arr, NUFFTGrid &nugrid, cudaStream_t stream) {
        Grid grid(arr.dims());
        filter_gradient_kernel <<<grid.blocks(), grid.threads(), 0, stream >>>(arr, nugrid.x, nugrid.y);
    }
} // namespace tomocam
